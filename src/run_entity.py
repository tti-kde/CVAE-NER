import json
import argparse
import os
import sys
import random
import time
from typing import List
from tqdm.contrib import tenumerate
import numpy as np
import re

from adjustText import adjust_text

from shared.data_structures import Dataset
from shared.const import (
    task_ner_labels,
    ner_labels_for_tsne,
    get_labelmap,
    general_label_map_other,
    color_list,
    aioner_original,
)
from shared.utils import convert_dataset_to_samples, NpEncoder, frange_cycle_linear
from main import initialize_model, initialize_optimizer, run_batch, make_label_embedding
from data_loader import SampleCollator, MultiDatasetSampler
from shared.bc8_data_reshape import output2bc8

from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
from accelerate import Accelerator
import accelerate.logging as logging
import itertools


def save_model(model, tokenizer, args):
    """
    Save the model to the output directory
    """
    args.logger.info("Saving model to %s..." % (args.output_dir))
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.encoder.save_pretrained(args.output_dir)
    torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "model_weights.bin"))
    tokenizer.save_pretrained(args.output_dir)


def output_ner_predictions(
    model, dataloader, dataset, output_file, labels_range, ner_id2label, other_idx=0
):
    """
    Save the prediction as a json file
    """
    args.logger.info("Outputting...")

    current_doc = {"doc_num": -1, "doc_key": 0, "sentences": ""}
    bc8_out_js = dataset.input_js
    dataloader = itertools.chain(dataloader, ["end"])
    for batch in dataloader:
        if batch != "end":
            output_dict = run_batch(
                batch=batch,
                args=args,
                model=model,
                num_ner_labels=num_ner_labels,
                corpus_labels_mask=corpus_labels_mask,
                mask_range=labels_range,
                training=False,
                accelerator=accelerator,
            )
            pred_ner = output_dict["pred_ner"]
        bc8_out_js, current_doc = output2bc8(
            batch,
            bc8_out_js,
            pred_ner,
            current_doc,
            ner_id2label,
            other_idx,
        )

    args.logger.info("Output predictions to %s.." % (output_file))
    with open(output_file, "w") as f:
        f.write("\n".join(json.dumps(doc, cls=NpEncoder) for doc in bc8_out_js))


def analize_ner_predictions(model, batches, dataset, output_file, labels_range):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    sent_num = 0
    count = 0

    for batch in batches:
        output_dict = run_batch(
            batch=batch,
            args=args,
            model=model,
            num_ner_labels=num_ner_labels,
            corpus_labels_mask=corpus_labels_mask,
            mask_range=labels_range,
            training=False,
            accelerator=accelerator,
        )
        pred_ner = output_dict["pred_ner"]
        ner_probs = output_dict["ner_probs"]
        for sample, preds, probs in zip(batch, pred_ner, ner_probs):
            off = sample["sent_start_in_doc"] - sample["sent_start"]
            k = sample["doc_key"] + "-" + str(sample["sentence_ix"])
            ner_result[k] = []
            for span, pred, prob in zip(sample["spans"], preds, probs):
                if span[0] == 0 and span[1] == 0:
                    sent_num = 0
                elif span[0] == span[1]:
                    count += 1
                    if count == 2:
                        sent_num += 1
                else:
                    count = 0
                prob_max = prob.max()
                if prob_max > 0.8:
                    if pred == 1:
                        ner_result[k].append([sent_num, span[0] + off, span[1] + off, pred, prob])

    js = dataset.js
    output_analize = [0] * len(js)
    for i, doc in enumerate(js):
        output = {"predicted_ner": []}
        for j in range(len(doc["sentences"])):
            k = doc["doc_key"] + "-" + str(j)
            sent_off = 0
            sent_num = 0
            if k in ner_result:
                for ner in ner_result[k]:
                    ne = doc["sentences"][ner[0]][ner[1] - sent_off : ner[2] - sent_off + 1]
                    if sent_num != ner[0]:
                        sent_off += len(doc["sentences"][ner[0]])
                    sent_num = ner[0]
                    ner[0] = ne
                    output["predicted_ner"].append(ner)
        output_analize[i] = output

    args.logger.info("Output predictions to %s.." % (output_file))
    with open(output_file, "w") as f:
        f.write("\n".join(json.dumps(doc, cls=NpEncoder) for doc in output_analize))


def _train(
    args: dict,
    num_ner_labels: int,
    others_idx: List = None,
    labels_range: List = None,
    corpus_name: List = None,
    accelerator=None,
    dev_data=None,
    train_data=None,
):
    if args.wandb is not None:
        accelerator.init_trackers(
            project_name=args.wandb[0],
            config=args_as_dict,
            init_kwargs={
                "wandb": {
                    "name": args.wandb[1],
                    "dir": args.output_dir,
                }
            },
        )
    args.device_count = accelerator.num_processes
    args.model_device = accelerator.device.type
    args, model, tokenizer, corpus_labels_mask = initialize_model(
        args, num_ner_labels, mask_range=labels_range
    )
    optimizer = initialize_optimizer(args, model)
    dev_samples, dev_ner, _, dev_num_spans = convert_dataset_to_samples(
        dev_data,
        tokenizer,
        ner_label2id=args.ner_label2id_valid,
        args=args,
    )
    train_samples, _, _, _ = convert_dataset_to_samples(
        train_data,
        tokenizer,
        ner_label2id=args.ner_label2id_train,  # [task],
        disease_limit=args.disease_limit,
        args=args,
    )
    num_each_train_datasets = [len(train_samples)]
    if corpus_name[1] == "bigbio":
        for i, dataset in enumerate(bigbio_datasets):
            extra_train_data = load_from_bigbio(dataset)
            extra_train_samples, _, _, _ = convert_dataset_to_samples(
                extra_train_data,
                tokenizer,
                ner_label2id=args.ner_label2id_train,  # [task],
                data_num=i + 1,
                others_idx=others_idx,
                args=args,
            )
    elif corpus_name[1] == "aioner_original":
        for i, dataset in enumerate(args.aio_dataset):
            extra_train_data = os.path.join(args.data_dir, "aioner", f"pubtator/{dataset}.json")
            extra_train_data = Dataset(extra_train_data, args=args, bool_bc8=True)
            extra_train_samples, _, _, _ = convert_dataset_to_samples(
                extra_train_data,
                tokenizer,
                ner_label2id=args.ner_label2id_train,
                data_num=i + 1,
                others_idx=others_idx,
                args=args,
            )
            train_samples += extra_train_samples
            num_each_train_datasets.append(len(extra_train_samples))
            if args.num_aioner == i + 1:
                break

    args.logger.info(
        f"lr: {args.learning_rate} task_lr: {args.task_learning_rate} alpha: {args.alpha}"
    )
    sample_collator = SampleCollator(tokenizer)
    multi_dataset_sampler = MultiDatasetSampler(
        num_each_train_datasets,
        args.train_batch_size,
        args.train_shuffle,
        args.dataset_batch_size,
        limit_num_samples=args.limit_num_samples,
    )
    train_dataloader = DataLoader(
        train_samples,
        batch_size=args.train_batch_size,
        collate_fn=sample_collator,
        sampler=multi_dataset_sampler,
    )
    dev_dataloader = DataLoader(
        dev_samples,
        shuffle=False,
        batch_size=args.eval_batch_size,
        collate_fn=sample_collator,
    )
    t_total = len(train_dataloader) * args.num_epoch
    eval_step = (len(train_dataloader) + args.device_count - 1) // (
        args.eval_per_epoch * args.device_count
    )
    if args.beta_warmup:
        beta_list = frange_cycle_linear(t_total // 2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(t_total * args.warmup_proportion), t_total
    )
    model, optimizer, scheduler, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, dev_dataloader
    )
    tr_loss = 0
    if args.each_dataset_loss:
        tr_loss_dict = {task: 0 for task in args.task_list[: args.num_aioner + 1]}
    global_step = 0
    best_result = 0.0
    best_epoch = 0
    count = 0
    for epoch in range(args.num_epoch):
        for i, batch in tenumerate(train_dataloader):
            with accelerator.accumulate(model):
                if args.beta_warmup:
                    args.beta = beta_list[global_step]
                output_dict = run_batch(
                    batch=batch,
                    args=args,
                    model=model,
                    num_ner_labels=num_ner_labels,
                    corpus_labels_mask=corpus_labels_mask,
                    mask_range=labels_range,
                    training=True,
                    accelerator=accelerator,
                )
                loss = output_dict["ner_loss"]
                if not args.simple and args.do_train:
                    class_loss = output_dict["class_loss"]
                    cvae_loss = output_dict["cvae_loss"]
                accelerator.wait_for_everyone()
                accelerator.backward(loss)

                if args.wandb is not None:
                    accelerator.log({"train/loss": loss.item()})
                    if args.entity_const:
                        accelerator.log(
                            {"train/entity_const_loss": output_dict["entity_const_loss"].item()}
                        )
                    if not args.simple:
                        accelerator.log({"train/class_loss": class_loss.item()})
                        accelerator.log({"train/cvae_loss": cvae_loss.item()})
                tr_loss += loss.item()
                if args.each_dataset_loss:
                    tr_loss_dict[batch["corpus_name"][0]] += loss.item()

                global_step += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (global_step * args.device_count) % (
                args.print_loss_step * args.gradient_accumulation_steps
            ) == 0:
                args.logger.info("Epoch=%d, iter=%d, loss=%.5f" % (epoch, i, tr_loss))
                if args.wandb is not None:
                    accelerator.log({"train/tr_loss": tr_loss})
                tr_loss = 0
                if args.each_dataset_loss:
                    for task in args.task_list[: args.num_aioner + 1]:
                        args.logger.info(f"{task} loss: {tr_loss_dict[task]}")
                        if args.wandb is not None:
                            accelerator.log({f"train/{task}_loss": tr_loss_dict[task]})
                        tr_loss_dict[task] = 0

            if global_step % eval_step == 0:
                torch.cuda.empty_cache()
                p, r, f1, _, _, _, _, _ = evaluate(
                    model,
                    dev_dataloader,
                    dev_ner,
                    labels_range,
                    corpus_labels_mask=corpus_labels_mask,
                    accelerator=accelerator,
                    num_spans=dev_num_spans,
                )
                yield f1
                if f1 > best_result:
                    best_result = f1
                    best_epoch = epoch
                    args.logger.info("!!! Best valid (epoch=%d): %.2f" % (epoch, f1 * 100))
                    save_model(model, tokenizer, args)
                    count = 0
                else:
                    count += 1

                if args.wandb is not None:
                    accelerator.log(
                        {
                            "dev/precision": p,
                            "dev/recall": r,
                            "dev/f1": f1,
                            "epoch": epoch,
                            "dev/best_score": best_result,
                            "dev/best_epoch": best_epoch,
                        }
                    )
                torch.cuda.empty_cache()
                if args.early_stopping is not None:
                    if count == args.early_stopping and best_result != 0.0:
                        args.logger.info("Early Stopping!!!")
                        break
        else:
            continue
        if args.wandb is not None:
            accelerator.end_training()
        break


def train(
    args: dict,
    num_ner_labels: int,
    others_idx: List = None,
    labels_range: List = None,
    corpus_name: List = None,
    accelerator=None,
    dev_data=None,
    train_data=None,
):
    return list(
        _train(
            args,
            num_ner_labels,
            others_idx,
            labels_range,
            corpus_name,
            accelerator,
            dev_data,
            train_data,
        )
    )


def evaluate(
    model,
    batches,
    tot_gold,
    labels_range,
    target_label=None,
    label2id=None,
    confu_matrix=False,
    dataset_num=0,
    tot_pred_list=None,
    tot_gold_list=None,
    return_exp=False,
    corpus_labels_mask=None,
    accelerator=None,
    num_spans=None,
    labels=None,
):
    """
    Evaluate the entity model
    """
    args.logger.info("Evaluating...")
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    pred_array = 0
    spans_embedding = 0
    mu_exp = 0
    preds_list = []
    gold_list = []
    tot_num_gold = 0
    if tot_pred_list is None:
        tot_pred_list = [0] * len(batches)
        tot_gold_list = [0] * len(batches)
    if dataset_num != 0 and target_label is not None:
        target_label = f"{target_label}:{dataset_num}"
    if return_exp and args.num_instance is not None:
        num_ne = (
            [0] * (len(labels) + 1)
            if args.special_ne is None
            else [0] * (len(labels) + 1 + len(args.special_ne))
        )

    for i, batch in enumerate(batches):
        if type(tot_pred_list[i]) == int:
            output_dict = run_batch(
                batch=batch,
                args=args,
                model=model,
                num_ner_labels=num_ner_labels,
                corpus_labels_mask=corpus_labels_mask,
                mask_range=labels_range,
                training=False,
                accelerator=accelerator,
            )
            tot_pred_list[i] = output_dict["pred_ner"]
            if args.num_instance is None:
                tot_gold_list[i] = output_dict["target_ner"]
            pred_ner = output_dict["pred_ner"]
            target_ner = output_dict["target_ner"]
        else:
            pred_ner = tot_pred_list[i]
            target_ner = tot_gold_list[i]
        for pred, target in zip(pred_ner, target_ner):
            if l_tot == num_spans:
                break
            if target == -100:
                continue
            if labels_range[0] is not None:
                extra_others = labels_range[dataset_num][0]
                if pred == extra_others:
                    pred = 0
                if target == extra_others:
                    target = 0
            l_tot += 1
            if target_label is not None:
                if target != label2id[target_label]:
                    target = 0
                if pred != label2id[target_label]:
                    pred = 0
            if target != 0:
                tot_num_gold += 1
            if pred == target:
                l_cor += 1
            if pred != 0 and target != 0 and pred == target:
                cor += 1
            if pred != 0:
                tot_pred += 1
        if return_exp and args.num_instance is not None:
            first_instance = True
            range_labels = (
                range(len(labels) + 1)
                if args.special_ne is None
                else list(range(len(labels) + 1)) + list(range(100, 100 + len(args.special_ne)))
            )
            for j, label in enumerate(range_labels):
                if num_ne[j] >= args.num_instance:
                    continue
                if args.remove_other and label == 0:
                    continue
                if labels_range[0] is not None and label < 100:
                    label = labels_range[dataset_num][0] + label
                spans_embedding_label = output_dict["spans_embedding"][target_ner == label]
                spans_embedding = (
                    spans_embedding_label
                    if i == 0 and first_instance
                    else torch.concatenate([spans_embedding, spans_embedding_label])
                )
                tot_gold_list[i] = (
                    target_ner[target_ner == label]
                    if first_instance
                    else np.concatenate([tot_gold_list[i], target_ner[target_ner == label]])
                )
                num_ne[j] += spans_embedding_label.shape[0]
                first_instance = False
            if type(tot_gold_list[i]) == int:
                tot_gold_list = tot_gold_list[:i]
                break
        elif return_exp:
            spans_embedding = (
                output_dict["spans_embedding"][target_ner != -100]
                if i == 0
                else torch.concatenate(
                    [
                        spans_embedding,
                        output_dict["spans_embedding"][target_ner != -100],
                    ]
                )
            )
    if confu_matrix:
        confu_matrix = confusion_matrix(gold_list, preds_list)
        sns.heatmap(confu_matrix, annot=True, cmap="Blues")
        plt.savefig(os.path.join(args.output_dir, "confu_matrix.png"))
    acc = l_cor / l_tot
    args.logger.info(f"total gold: {tot_num_gold}")
    args.logger.info("Accuracy: %5f" % acc)
    args.logger.info("Cor: %d, Pred TOT: %d, Gold TOT: %d" % (cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    args.logger.info("P: %.5f, R: %.5f, F1: %.5f" % (p, r, f1))
    args.logger.info(f"{p}\t{r}\t{f1}")
    args.logger.info("Used time: %f" % (time.time() - c_time))
    return p, r, f1, pred_array, spans_embedding, mu_exp, tot_pred_list, tot_gold_list


def mesure_label_distance(label_embedding, args):
    label_name = []
    for task in args.task_list:
        label_name += [f"other_{task}"] + [f"{label}_{task}" for label in task_ner_labels[task]]
    label_embedding = label_embedding / np.linalg.norm(label_embedding, axis=1).reshape(-1, 1)
    label_distance = np.dot(label_embedding, label_embedding.T)
    np.fill_diagonal(label_distance, 0)
    label_distance = np.arccos(label_distance) / np.pi
    label_distance = 1 - label_distance
    np.fill_diagonal(label_distance, 0)
    args.logger.info(label_distance.shape)
    args.logger.info(label_distance)
    for dis, label in zip(label_distance, label_name):
        unsorted_min_indices = np.argpartition(dis, 6)[:6]
        y = dis[unsorted_min_indices]
        indices = unsorted_min_indices[np.argsort(y)[1:]]
        args.logger.info(f"{label}: {[label_name[i] for i in indices]}")
    args.logger.info("under half with BioRED label")
    pattern = "bc8"
    for dis, label in zip(label_distance, label_name):
        unsorted_max_indices = np.argpartition(-dis, 10)[:10]
        y = dis[unsorted_max_indices]
        indices = unsorted_max_indices[np.argsort(-y)]
        args.logger.info(
            f"{label}: {[label_name[i] for i in indices if re.search(pattern, label_name[i])]}"
        )
    args.logger.info("top with BioRED label")
    pattern = "bc8"
    for dis, label in zip(label_distance, label_name):
        unsorted_max_indices = np.argpartition(dis, 10)[:10]
        y = dis[unsorted_max_indices]
        indices = unsorted_max_indices[np.argsort(y)]
        args.logger.info(
            f"{label}: {[label_name[i] for i in indices if re.search(pattern, label_name[i])]}"
        )


def make_label_tsne_graph(x, args):
    # t-SNE
    tsne = TSNE(n_components=2, init="random", random_state=0, perplexity=args.tsne_perplexity)
    out_file_name = "label.png"
    args.logger.info(x.shape)
    args.logger.info(x)
    span_tsne = tsne.fit_transform(x)
    args.logger.info(span_tsne.shape)
    args.logger.info(span_tsne)
    others_list = []
    y_test = []
    for task in args.task_list:
        y_test += [f"other_{task}"] + ner_labels_for_tsne[task]
        others_list += [f"other_{task}"]
    plt.xlim(span_tsne[:, 0].min() - 10, span_tsne[:, 0].max() + 20)
    plt.ylim(span_tsne[:, 1].min() - 10, span_tsne[:, 1].max() + 10)
    plt.clf()
    i = 0
    texts = []
    for name, label in zip(y_test, span_tsne):
        if args.remove_other and name in others_list:
            continue
        plt.scatter(
            label[0],
            label[1],
            label=name,
            color=color_list[i],
        )
        if i < 6:
            texts += [plt.text(label[0], label[1], name, fontsize=8)]
        i += 1
    adjust_text(texts)
    plt.xlabel("t-SNE Feature1")
    plt.ylabel("t-SNE Feature2")
    plt.savefig(os.path.join(args.output_dir, out_file_name))


def make_tsne_graph(y, x, bool_mu=False, target_label=None, others_idx=None):
    # t-SNE
    tsne = TSNE(n_components=2, init="random", random_state=0, perplexity=args.tsne_perplexity)
    out_file_name = "mu" if bool_mu else "instance"
    if target_label is not None:
        if args.special_ne is not None:
            target_label += list(range(100, 100 + len(args.special_ne)))
        mask = False
        for i, label in enumerate(target_label):
            mask = (y == label) | mask
        x = x[mask]
        y = y[mask]
    args.logger.info(x.shape)
    args.logger.info(y.shape)
    assert x.shape[0] == y.shape[0]
    span_tsne = tsne.fit_transform(x)
    args.logger.info(span_tsne.shape)
    y_test = []
    for task in args.task_list:
        y_test += [f"other_{task}"] + ner_labels_for_tsne[task]
    if args.case_tsne is not None:
        y_test += args.case_tsne
    plt.xlim(span_tsne[:, 0].min() - 10, span_tsne[:, 0].max() + 20)
    plt.ylim(span_tsne[:, 1].min() - 10, span_tsne[:, 1].max() + 10)
    dot_size = 20
    print(target_label)
    labels = np.unique(y)
    print(labels)
    if args.special_ne is None:
        plt.clf()
        for i, each_label in enumerate(labels):
            if each_label >= 500:
                each_label = len(y_test) - 1
            c_plot_bool = y == each_label
            plt.scatter(
                span_tsne[c_plot_bool, 0],
                span_tsne[c_plot_bool, 1],
                label=y_test[each_label],
                color=color_list[i],
                s=dot_size,
            )
        plt.xlabel("t-SNE Feature1")
        plt.ylabel("t-SNE Feature2")
        plt.legend(
            bbox_to_anchor=(0.9, 1.04),
            loc="upper left",
            borderaxespad=0,
            fontsize="xx-small",
        )
        plt.savefig(os.path.join(args.output_dir, f"{out_file_name}.png"))
    for biored_label in range(1, 7):
        if (
            args.special_ne is not None
            and target_label is not None
            and biored_label not in target_label
        ):
            continue
        biored_handle_label = [biored_label]
        for idx, dataset in enumerate(args.aio_dataset):
            biored_handle_label += [
                k + others_idx[idx + 1]
                for k, v in general_label_map_other[dataset].items()
                if v == biored_label
            ]
        plt.clf()
        fig, ax = plt.subplots()
        texts = []
        marker_list = ["o", "x", "^"]

        color_index = 0
        labels = sorted(labels, key=lambda x: 1 if x in [4, 45] else 0)
        for i, each_label in enumerate(labels):
            c_plot_bool = y == each_label
            if span_tsne[c_plot_bool, :].shape[0] == 0:
                continue
            if each_label in biored_handle_label:
                color = color_list[color_index]
                if each_label == 19:
                    color = "blue"
                elif each_label == 20:
                    color = "green"
                elif each_label == 21 or each_label == 22 or each_label == 38:
                    continue
                color_index += 1
                ax.scatter(
                    span_tsne[c_plot_bool, 0],
                    span_tsne[c_plot_bool, 1],
                    label=y_test[each_label],
                    color=color,
                    s=dot_size,
                )
            elif each_label >= 100:
                ax.scatter(
                    span_tsne[c_plot_bool, 0],
                    span_tsne[c_plot_bool, 1],
                    color="black",
                    marker="x",
                    s=dot_size,
                )
            else:
                ax.scatter(
                    span_tsne[c_plot_bool, 0],
                    span_tsne[c_plot_bool, 1],
                    color="#D3D3D3",
                    s=dot_size,
                )
        if len(texts) != 0:
            adjust_text(
                texts,
                force_points=(0.1, 0.2),  # 点からの反発力
                force_text=(0.02, 0.1),  # テキスト間の反発力
                expand_points=(0.1, 0.1),  # 点からの最小距離
                expand_text=(0.05, 0.05),  # テキスト間の最小距離
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
                ax=ax,
            )

        le_position = (0.65, 1.1) if args.special_ne is None else (0.66, 0.25)
        ax.set_xlabel("t-SNE Feature1")
        ax.set_xlabel("t-SNE Feature2")
        ax.legend(
            bbox_to_anchor=le_position,
            loc="upper left",
            borderaxespad=0,
            fontsize="small",
            framealpha=1.0,
        )
        plt.savefig(os.path.join(args.output_dir, f"{out_file_name}_{biored_label}.png"))


def load_from_bigbio(dataset, doc_range=None, dev_bigbio=False):
    if dataset == "genetag":
        dataset_tag = dataset + "correct"
    elif dataset == "medmentions":
        dataset_tag = dataset + "_full"
    elif dataset == "chebi_nactem":
        dataset_tag = dataset + "_abstr_ann1"
    elif dataset == "distemist":
        dataset_tag = dataset + "_entities"
    elif dataset == "ctebmsp":
        dataset_tag = dataset + "_abstracts"
    elif dataset == "spl_adr_200db":
        dataset_tag = dataset + "_train"
    else:
        dataset_tag = dataset
    if dev_bigbio:
        data_type = "validation"
        if (
            dataset == "hprd50"
            or dataset == "mirna"
            or dataset == "bioinfer"
            or dataset == "tmvar_v1"
        ):
            data_type = "test"
        elif dataset == "chebi_nactem":
            data_type = "train"
            dataset_tag = "chebi_nactem_fullpaper"
    else:
        data_type = "train"
    if dataset == "progene":
        extra_train_data = load_dataset(
            "bigbio/" + dataset,
            name=f"{dataset_tag}_bigbio_kb",
            split=f"split_0_{data_type}",
        )
        extra_train_data = Dataset(
            extra_train_data, doc_range=doc_range, bool_bigbio=True, args=args
        )
    else:
        extra_train_data = load_dataset("bigbio/" + dataset, name=f"{dataset_tag}_bigbio_kb")
        extra_train_data = Dataset(
            extra_train_data[f"{data_type}"],
            doc_range=doc_range,
            bool_bigbio=True,
            args=args,
        )
    return extra_train_data


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_list",
        type=str,
        required=True,
        nargs="*",
        help="'bc5cdr' 'ncbi' 'bc4chemdner'",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="path to the preprocessed dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="entity_output",
        help="output directory of the entity model",
    )

    parser.add_argument(
        "--max_span_length",
        type=int,
        default=8,
        help="spans w/ length up to max_span_length are considered as candidates",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="batch size during training"
    )
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="batch size during inference"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate for the BERT encoder",
    )
    parser.add_argument(
        "--task_learning_rate",
        type=float,
        default=1e-4,
        help="learning rate for task-specific parameters, i.e., classification head",
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.1,
        help="the ratio of the warmup steps to the total steps",
    )
    parser.add_argument("--num_epoch", type=int, default=100, help="number of the training epochs")
    parser.add_argument(
        "--print_loss_step",
        type=int,
        default=100,
        help="how often logging the loss value during training",
    )
    parser.add_argument(
        "--eval_per_epoch",
        type=int,
        default=1,
        help="how often evaluating the trained model on dev set during training",
    )
    parser.add_argument(
        "--bertadam",
        action="store_true",
        help="If bertadam, then set correct_bias = False",
    )

    parser.add_argument("--do_train", action="store_true", help="whether to run training")
    parser.add_argument(
        "--train_shuffle",
        action="store_true",
        help="whether to train with randomly shuffled data",
    )
    parser.add_argument("--do_eval", action="store_true", help="whether to run evaluation")
    parser.add_argument("--bc8_valid", action="store_true")
    parser.add_argument("--eval_test", action="store_true", help="whether to evaluate on test set")
    parser.add_argument(
        "--dev_pred_filename",
        type=str,
        default="ent_pred_dev.json",
        help="the prediction filename for the dev set",
    )
    parser.add_argument(
        "--test_pred_filename",
        type=str,
        default="ent_pred_test.json",
        help="the prediction filename for the test set",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="the base model name (a huggingface model)",
    )
    parser.add_argument("--bert_model_dir", type=str, default=None, help="the base model directory")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--context_window",
        type=int,
        required=True,
        default=None,
        help="the context window size W for the entity model",
    )
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--target_label", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--leaving_rate", type=int, default=10)
    parser.add_argument("--wandb", type=str, default=None, nargs="*")
    parser.add_argument("--dev_bigbio", type=str, default=None, nargs="*")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--type_tsne", type=str, default=None)
    parser.add_argument("--tsne_doc_num", type=int, default=None)
    parser.add_argument("--num_instance", type=int, default=None)
    parser.add_argument("--remove_other", action="store_true")
    parser.add_argument("--special_ne", type=lambda s: s.split(","), default=None)
    parser.add_argument("--cuda_num", type=int, default=None)
    parser.add_argument("--eval_train_data", action="store_true")
    parser.add_argument("--early_stopping", type=int, default=None)
    parser.add_argument("--optuna", type=str, help="study_name of optuna")
    parser.add_argument("--optuna_storage", type=str, help="Storage for optuna")
    parser.add_argument("--optuna_n_trials", type=int, default=100, help="n_trials for optuna")
    parser.add_argument("--disease_limit", type=int, default=None)
    parser.add_argument(
        "--optimize_option", type=str, default=None, choices=["same", "lr_sep", "alt"]
    )
    parser.add_argument("--label_stopping_epoch", type=int, default=20)
    parser.add_argument("--not_corpus_difference", action="store_true")
    parser.add_argument("--corpus_one_hot", action="store_true")
    parser.add_argument("--general_one_hot", type=str, default=None, choices=["other", "wo_other"])
    parser.add_argument("--eval_extra_data", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--bc8_train", action="store_true")
    parser.add_argument("--corpus_token_pos", type=str, default=None)
    parser.add_argument("--label_token", action="store_true")
    parser.add_argument("--lookup_label_token", action="store_true")
    parser.add_argument("--lookup_label_token_onehot", type=str, default=None)
    parser.add_argument("--shared_label_token", action="store_true")
    parser.add_argument("--output_prediction", action="store_true")
    parser.add_argument("--hidden_exp_cvae", action="store_true")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--bigbio_dir", type=str, default="./bigbio.txt")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_warmup", action="store_true")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dynamic_mu", type=str, default=None)
    parser.add_argument("--condition_dynamic_mu", action="store_true")
    parser.add_argument("--saved_label_embedding_dir", type=str, default=None)
    parser.add_argument("--save_label_embedding", type=str, default=None)
    parser.add_argument("--condition_freezing", type=bool, default=True)
    parser.add_argument("--label_emb_input", type=str, default="label")
    parser.add_argument("--label_emb_cls", action="store_true")
    parser.add_argument("--enc_hidden_dim", type=int, default=300)
    parser.add_argument("--z_hidden_dim", type=int, default=150)
    parser.add_argument("--tsne_label", type=int, default=None, nargs="*")
    parser.add_argument("--case_tsne", type=str, default=None, nargs="*")
    parser.add_argument("--range_bigbio", type=int, default=None, nargs="*")
    parser.add_argument("--doc_range", type=int, default=[0, 100], nargs="*")
    parser.add_argument("--selected_bigbio", type=int, default=None, nargs="*")
    parser.add_argument("--limit_num_samples", action="store_true")
    parser.add_argument("--entity_const", action="store_true")
    parser.add_argument("--entity_const_weight", type=float, default=10)
    parser.add_argument("--entity_const_th", type=float, default=0.5)
    parser.add_argument("--aioner_test", action="store_true")
    parser.add_argument("--use_t5", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_reft", action="store_true")
    parser.add_argument("--bool_dora", action="store_true")
    parser.add_argument("--bool_quantize", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_aioner", type=int, default=9)
    parser.add_argument("--each_dataset_loss", action="store_true")

    args = parser.parse_args()
    args_as_dict = vars(args)
    accelerator = Accelerator(
        log_with="wandb" if args.wandb is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    stream_handler = logging.logging.StreamHandler()
    stream_handler.setLevel(logging.logging.INFO)
    if args.do_train:
        file_handler = logging.logging.FileHandler(os.path.join(args.output_dir, "train.log"), "w")
    elif args.eval_extra_data:
        file_handler = logging.logging.FileHandler(
            os.path.join(args.output_dir, "extra_data_num.log"), "w"
        )
    elif args.aioner_test:
        file_handler = logging.logging.FileHandler(
            os.path.join(args.output_dir, "aioner_test.log"), "w"
        )
    else:
        if args.eval_test:
            file_handler = logging.logging.FileHandler(
                os.path.join(args.output_dir, "eval.log"), "w"
            )
        else:
            file_handler = logging.logging.FileHandler(
                os.path.join(args.output_dir, "eval_dev.log"), "w"
            )
    file_handler.setLevel(logging.logging.INFO)
    logging.logging.basicConfig(
        handlers=[stream_handler, file_handler],
    )
    args.logger = logging.get_logger("root", log_level="DEBUG")

    args.bool_bc8 = args.task_list[0] == "bc8"
    if args.task_list[0] == "aioner_original":
        args.task_list = [list(aioner_original.keys())[args.num_aioner]]
        args.train_data = args.dev_data = os.path.join(
            args.data_dir, "aioner/pubtator", f"{args.task_list[0]}.json"
        )
        args.split = 100
        args.bool_bc8 = True
    elif args.split == 0:
        args.train_data = os.path.join(args.data_dir, args.task_list[0], "json/train.json")
        args.dev_data = os.path.join(args.data_dir, args.task_list[0], "json/dev.json")
    else:
        args.train_data = args.dev_data = os.path.join(
            args.data_dir, args.task_list[0], "json/train.json"
        )
    if args.eval_test:
        args.test_data = os.path.join(args.data_dir, args.task_list[0], "json/test.json")

    if args.wandb is not None:
        wandb.login(key="wandb_key")

    setseed(args.seed)

    args.logger.info(sys.argv)
    args.logger.info(args)

    task = args.task_list[0]
    labels = task_ner_labels[task]
    args.ner_label2id_valid, ner_id2label_valid = get_labelmap(labels)
    num_ner_labels = len(labels) + 1
    args.ner_label2id_train = args.ner_label2id_valid

    labels_range = [None, None]
    others_idx = None
    corpus_name = args.task_list if len(args.task_list) > 1 else [None, None]
    if corpus_name[1] == "bigbio":
        f = open(args.bigbio_dir)
        bigbio_datasets = f.read().splitlines()
        f.close()
        if args.range_bigbio is not None:
            bigbio_datasets = bigbio_datasets[args.range_bigbio[0] : args.range_bigbio[1]]
            args.logger.info(f"BigBio corpus: {bigbio_datasets}")
        elif args.selected_bigbio is not None:
            temp_bigbio_datasets = []
            for data_num in args.selected_bigbio:
                temp_bigbio_datasets.append(bigbio_datasets[data_num])
            bigbio_datasets = temp_bigbio_datasets
            args.logger.info(f"BigBio corpus: {bigbio_datasets}")
        args.task_list = [args.task_list[0]] + bigbio_datasets
    elif corpus_name[1] == "aioner_original":
        args.aio_dataset = list(aioner_original.keys())
        args.task_list = [args.task_list[0]] + args.aio_dataset
    if args.not_corpus_difference:
        others_idx = [0, num_ner_labels]
        labels_range = [(others_idx[0], others_idx[1])]
        for i in range(1, len(args.task_list)):
            label_map = general_label_map_other[args.task_list[i]]
            dataset_label = task_ner_labels[args.task_list[i]]
            add_label2id = {
                f"{dataset_label[label - 1]}:{i}": label_biored
                for label, label_biored in label_map.items()
                if label != 0
            }
            labels_range_dataset = set(add_label2id.values())
            args.ner_label2id_train.update(add_label2id)
            labels_range.append(list(labels_range_dataset))
        args.logger.info(args.ner_label2id_train)
        args.logger.info(len(args.ner_label2id_train))
    elif (
        len(args.task_list) > 1
        or args.label_token
        or args.shared_label_token
        or args.lookup_label_token_onehot
    ):
        others_idx = [0, num_ner_labels]
        labels_range = [(others_idx[0], others_idx[1])]
        for i in range(1, len(args.task_list)):
            labels_train = [f"{label}:{i}" for label in task_ner_labels[args.task_list[i]]]
            add_label2id, _ = get_labelmap(labels_train, initial_value=num_ner_labels)
            args.ner_label2id_train.update(add_label2id)
            num_ner_labels += len(task_ner_labels[args.task_list[i]]) + 1
            others_idx.append(num_ner_labels)
            labels_range.append((others_idx[i], others_idx[i + 1]))
            if (
                args.num_aioner == i
                and args.save_label_embedding is None
                and args.type_tsne is None
            ):
                break
        args.ner_id2label_train = {v: k for k, v in args.ner_label2id_train.items()}

    if args.do_train:
        dev_data = Dataset(args.dev_data, bool_bc8=args.bool_bc8, args=args)
        train_data = Dataset(args.train_data, bool_bc8=args.bool_bc8, args=args)
        if args.optuna:

            def objective(trial):
                # search space
                args.learning_rate = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
                args.task_learning_rate = trial.suggest_float("task_lr", 1e-4, 1e-3, log=True)
                # args.alpha = trial.suggest_int("alpha", 1, 1000, log=True)

                max_score = -float("inf")
                for step, score in enumerate(
                    _train(
                        args,
                        num_ner_labels,
                        others_idx,
                        labels_range,
                        corpus_name,
                        accelerator,
                        dev_data,
                        train_data,
                    )
                ):
                    max_score = max(max_score, score)
                    trial.report(score, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return max_score

            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource="auto", reduction_factor=4, min_early_stopping_rate=0
            )
            study = optuna.create_study(
                study_name=args.optuna,
                storage=args.optuna_storage,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
            study.optimize(objective, n_trials=args.optuna_n_trials)

            # if without SQL, output tsv
            # if not args.optuna_storage:
            #     study.trials_dataframe().to_csv(args.optuna + ".tsv", sep="\t")
        else:
            train(
                args,
                num_ner_labels,
                others_idx,
                labels_range,
                corpus_name,
                accelerator,
                dev_data,
                train_data,
            )

    if args.do_eval:
        args.bert_model_dir = args.output_dir
        args.device_count = accelerator.num_processes
        args.model_device = accelerator.device.type
        args, model, tokenizer, corpus_labels_mask = initialize_model(
            args, num_ner_labels, mask_range=labels_range
        )
        if args.eval_test and args.bc8_valid:
            test_data = Dataset(args.dev_data, bool_bc8=args.bool_bc8, args=args)
            prediction_file = os.path.join(args.output_dir, "ent_pred_bc8_test.json")
        elif args.eval_test:
            test_data = Dataset(args.test_data, bool_bc8=args.bool_bc8, args=args)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        elif args.bc8_valid:
            test_data = Dataset(args.dev_data, bool_bc8=args.bool_bc8, args=args)
            prediction_file = os.path.join(args.output_dir, "ent_pred_bc8_valid.json")
        else:
            test_data = Dataset(args.dev_data, bool_bc8=args.bool_bc8, args=args)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner, each_test_ner, test_num_spans = convert_dataset_to_samples(
            test_data,
            tokenizer,
            ner_label2id=args.ner_label2id_valid,
            labels=labels,
            args=args,
        )
        sample_collator = SampleCollator(tokenizer)
        test_dataloader = DataLoader(
            test_samples,
            shuffle=False,
            batch_size=args.eval_batch_size,
            collate_fn=sample_collator,
        )
        model, test_dataloader = accelerator.prepare(model, test_dataloader)  # , test_dataloader
        if args.bc8_valid:
            output_ner_predictions(
                model,
                test_dataloader,
                test_data,
                output_file=prediction_file,
                labels_range=labels_range,
                ner_id2label=ner_id2label_valid,
            )
        elif args.eval_train_data:
            train_data = Dataset(args.train_data, bool_bc8=args.bool_bc8, args=args)
            train_samples, train_ner, each_train_ner, _ = convert_dataset_to_samples(
                train_data,
                tokenizer,
                ner_label2id=args.ner_label2id_train,
                labels=labels,
                disease_limit=args.disease_limit,
                args=args,
            )
        else:
            (
                _,
                _,
                _,
                pred_array_bc,
                spans_exp_bc,
                mu_exp,
                tot_pred_list,
                tot_gold_list,
            ) = evaluate(
                model,
                test_dataloader,
                test_ner,
                labels_range,
                corpus_labels_mask=corpus_labels_mask,
                accelerator=accelerator,
                num_spans=test_num_spans,
            )
            if args.target_label:
                for target_label in task_ner_labels[args.task_list[0]]:
                    args.logger.info(f"Target_label: {target_label}")
                    evaluate(
                        model,
                        test_dataloader,
                        each_test_ner[target_label],
                        labels_range,
                        target_label=target_label,
                        label2id=args.ner_label2id_valid,
                        corpus_labels_mask=corpus_labels_mask,
                        tot_pred_list=tot_pred_list,
                        tot_gold_list=tot_gold_list,
                        accelerator=accelerator,
                        num_spans=test_num_spans,
                    )
            if args.output_prediction:
                if (
                    args.dev_pred_filename == "analize.json"
                    or args.test_pred_filename == "analize.json"
                ):
                    analize_ner_predictions(
                        model,
                        test_dataloader,
                        test_data,
                        output_file=prediction_file,
                        labels_range=labels_range,
                    )
                else:
                    output_ner_predictions(
                        model,
                        test_dataloader,
                        test_data,
                        output_file=prediction_file,
                        labels_range=labels_range,
                        ner_id2label=ner_id2label_valid,
                    )
            if args.tsne:
                args.logger.info("make tsne graph")
                make_tsne_graph(pred_array_bc, spans_exp_bc)
                if not args.simple:
                    make_tsne_graph(pred_array_bc, mu_exp, bool_mu=True)
    if args.eval_extra_data:
        args.bert_model_dir = args.output_dir
        args, model, tokenizer, corpus_labels_mask = initialize_model(
            args, num_ner_labels, mask_range=labels_range
        )
        extra_dev_batches = []
        for i, dataset in enumerate(bigbio_datasets):
            extra_dev_data = load_from_bigbio(dataset)
            (
                extra_dev_samples,
                extra_dev_ner,
                each_extra_train_ner,
            ) = convert_dataset_to_samples(
                extra_dev_data,
                tokenizer,
                ner_label2id=args.ner_label2id_train,
                data_num=i + 1,
                others_idx=others_idx,
                labels=task_ner_labels[args.task_list[1]],
                args=args,
            )
        args.logger.info(
            f"labels_range: {labels_range[1]}\nner_label2id: {args.ner_label2id_train}"
        )
        args.logger.info(f"ne_num: {each_extra_train_ner}")
    if args.aioner_test:
        args.bert_model_dir = args.output_dir
        args.device_count = accelerator.num_processes
        args.model_device = accelerator.device.type
        args, model, tokenizer, corpus_labels_mask = initialize_model(
            args, num_ner_labels, mask_range=labels_range
        )
        sample_collator = SampleCollator(tokenizer)
        for i, dataset in enumerate(args.aio_dataset):
            args.logger.info(f"Dataset: {dataset}")
            dataset_test = re.sub("TrainDev", "Test", dataset)
            dataset_test = re.sub("Train", "Test", dataset_test)
            extra_test_data = os.path.join(args.data_dir, "aioner", f"pubtator/{dataset_test}.json")
            extra_test_data = Dataset(extra_test_data, args=args, bool_bc8=True)
            extra_test_samples, extra_test_ner, each_test_ner, test_num_spans = (
                convert_dataset_to_samples(
                    extra_test_data,
                    tokenizer,
                    ner_label2id=args.ner_label2id_train,
                    data_num=i + 1,
                    labels=task_ner_labels[dataset],
                    others_idx=others_idx,
                    args=args,
                )
            )
            extra_test_dataloader = DataLoader(
                extra_test_samples,
                shuffle=False,
                batch_size=args.eval_batch_size,
                collate_fn=sample_collator,
            )
            model, extra_test_dataloader = accelerator.prepare(model, extra_test_dataloader)
            if args.output_prediction:
                if not os.path.exists(os.path.join(args.output_dir, f"output_pred")):
                    os.makedirs(os.path.join(args.output_dir, f"output_pred"))
                output_ner_predictions(
                    model,
                    extra_test_dataloader,
                    extra_test_data,
                    output_file=os.path.join(
                        args.output_dir, f"output_pred/{dataset_test}_pred.json"
                    ),
                    labels_range=labels_range,
                    ner_id2label=args.ner_id2label_train,
                    other_idx=others_idx[i + 1],
                )
            _, _, _, _, _, _, tot_pred_list, tot_gold_list = evaluate(
                model,
                extra_test_dataloader,
                extra_test_ner,
                labels_range,
                dataset_num=i + 1,
                # return_exp=True,
                corpus_labels_mask=corpus_labels_mask,
                accelerator=accelerator,
                num_spans=test_num_spans,
            )
            if args.target_label:
                for target_label in task_ner_labels[dataset]:
                    args.logger.info(f"Target_label: {target_label}")
                    evaluate(
                        model,
                        extra_test_dataloader,
                        each_test_ner[target_label],
                        labels_range,
                        target_label=target_label,
                        label2id=args.ner_label2id_train,
                        corpus_labels_mask=corpus_labels_mask,
                        dataset_num=i + 1,
                        tot_pred_list=tot_pred_list,
                        tot_gold_list=tot_gold_list,
                        num_spans=test_num_spans,
                    )
    if args.type_tsne is not None:
        tsne_tensor_path = os.path.join(args.output_dir, "tsne_tensor")
        tensor_dir_name = f"{args.num_aioner}"
        if not os.path.exists(tsne_tensor_path):
            os.makedirs(tsne_tensor_path)
        if args.num_instance is not None:
            tensor_dir_name += f"{args.num_instance}"
        if args.tsne_doc_num is not None:
            tensor_dir_name += f"{args.tsne_doc_num}"
        if args.special_ne is not None:
            tensor_dir_name += f"{len(args.special_ne)}"
        tsne_np_path = os.path.join(tsne_tensor_path, f"{tensor_dir_name}.npy")
        tsne_tensor_path = os.path.join(tsne_tensor_path, f"{tensor_dir_name}.pt")
        if os.path.exists(tsne_tensor_path):
            args.logger.info(f"Load tsne_tensor from {tsne_tensor_path}")
            tsne_label_array = np.load(tsne_np_path)
            tsne_embedding_tensor = torch.load(tsne_tensor_path)
        else:
            args.logger.info("Calculating tsne_tensor")
            args.bert_model_dir = args.output_dir
            args.device_count = accelerator.num_processes
            args.model_device = accelerator.device.type
            args, model, tokenizer, corpus_labels_mask = initialize_model(
                args, num_ner_labels, mask_range=labels_range
            )
            train_data = Dataset(args.train_data, bool_bc8=args.bool_bc8, args=args)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
            dev_samples, dev_ner, each_dev_ner, dev_num_spans = convert_dataset_to_samples(
                train_data,
                tokenizer,
                ner_label2id=args.ner_label2id_valid,
                labels=labels,
                args=args,
            )
            sample_collator = SampleCollator(tokenizer)
            train_dataloader = DataLoader(
                dev_samples[: args.tsne_doc_num],
                shuffle=False,
                batch_size=args.eval_batch_size,
                collate_fn=sample_collator,
            )
            model, train_dataloader = accelerator.prepare(model, train_dataloader)
            (
                _,
                _,
                _,
                pred_array_bc,
                tsne_embedding_tensor,
                mu_exp,
                tot_pred_list,
                tot_gold_list,
            ) = evaluate(
                model,
                train_dataloader,
                dev_ner,
                labels_range,
                return_exp=True,
                corpus_labels_mask=corpus_labels_mask,
                accelerator=accelerator,
                num_spans=dev_num_spans,
                labels=task_ner_labels[args.task_list[0]],
            )
            for i, dataset in enumerate(args.aio_dataset):
                if args.num_aioner != 9 and i != args.num_aioner:
                    continue
                args.logger.info(f"Dataset: {dataset}")
                extra_test_data = os.path.join(args.data_dir, "aioner", f"pubtator/{dataset}.json")
                extra_test_data = Dataset(extra_test_data, args=args, bool_bc8=True)
                extra_test_samples, extra_test_ner, each_test_ner, test_num_spans = (
                    convert_dataset_to_samples(
                        extra_test_data,
                        tokenizer,
                        ner_label2id=args.ner_label2id_train,
                        data_num=i + 1,
                        labels=task_ner_labels[dataset],
                        others_idx=others_idx,
                        args=args,
                    )
                )
                extra_test_dataloader = DataLoader(
                    extra_test_samples[: args.tsne_doc_num],
                    shuffle=False,
                    batch_size=args.eval_batch_size,
                    collate_fn=sample_collator,
                )
                model, extra_test_dataloader = accelerator.prepare(model, extra_test_dataloader)
                (
                    _,
                    _,
                    _,
                    extra_pred_array_bc,
                    extra_spans_exp,
                    extra_mu_exp,
                    extra_tot_pred_list,
                    extra_tot_gold_list,
                ) = evaluate(
                    model,
                    extra_test_dataloader,
                    extra_test_ner,
                    labels_range,
                    return_exp=True,
                    dataset_num=i + 1,
                    corpus_labels_mask=corpus_labels_mask,
                    accelerator=accelerator,
                    num_spans=test_num_spans,
                    labels=task_ner_labels[dataset],
                )
                tsne_embedding_tensor = torch.concatenate([tsne_embedding_tensor, extra_spans_exp])
                tot_gold_list = [np.concatenate(tot_gold_list + extra_tot_gold_list)]
            tsne_embedding_tensor = tsne_embedding_tensor.cpu()
            tsne_label_array = tot_gold_list[0]
            np.save(tsne_np_path, tsne_label_array)
            torch.save(tsne_embedding_tensor, tsne_tensor_path)
        make_tsne_graph(
            tsne_label_array,
            tsne_embedding_tensor.numpy(),
            target_label=args.tsne_label,
            others_idx=others_idx,
        )
    if args.save_label_embedding is not None:
        if args.num_aioner != 9:
            dataset_name = list(aioner_original.keys())[args.num_aioner]
            args.train_data = args.dev_data = os.path.join(
                args.data_dir, "aioner/pubtator", f"{dataset_name}.json"
            )
            args.output_dir = os.path.join(args.output_dir, dataset_name)
            args.save_label_embedding = os.path.join(args.save_label_embedding, dataset_name)
            args.num_aioner += 1
        elif len(args.task_list) > 1:
            dataset_name = args.task_list[0]
            args.num_aioner = None
        else:
            dataset_name = args.task_list[0]
        if not os.path.exists(args.save_label_embedding):
            os.makedirs(args.save_label_embedding)
        args.bert_model_dir = args.output_dir
        args.device_count = accelerator.num_processes
        args.model_device = accelerator.device.type
        args, model, tokenizer, corpus_labels_mask = initialize_model(
            args, num_ner_labels, mask_range=labels_range
        )
        train_data = Dataset(args.train_data, bool_bc8=args.bool_bc8, args=args)
        train_samples, train_ner, each_train_ner, _ = convert_dataset_to_samples(
            train_data,
            tokenizer,
            ner_label2id=args.ner_label2id_train,
            labels=task_ner_labels[dataset_name],
            disease_limit=args.disease_limit,
            data_num=args.num_aioner if len(args.task_list) > 1 else None,
            others_idx=others_idx,
            args=args,
        )
        sample_collator = SampleCollator(tokenizer)
        dataloader = DataLoader(
            train_samples,
            shuffle=False,
            batch_size=args.eval_batch_size,
            collate_fn=sample_collator,
        )
        model, dataloader = accelerator.prepare(model, dataloader)
        tot_label_embedding = torch.zeros(num_ner_labels, args.hidden_size)
        tot_instance_num = torch.zeros(num_ner_labels)
        for batch in dataloader:
            label_embedding, isinstance_num = make_label_embedding(
                args, model, batch, num_ner_labels, corpus_labels_mask
            )
            tot_label_embedding += label_embedding.detach().clone().cpu()
            tot_instance_num += isinstance_num.detach().clone().cpu()
        tot_label_embedding /= tot_instance_num.unsqueeze(1)
        tot_label_embedding = torch.where(tot_instance_num.unsqueeze(1) > 0, tot_label_embedding, 0)
        if len(args.task_list) > 1:
            dataset_range = (
                [
                    others_idx[args.num_aioner],
                    others_idx[args.num_aioner + 1],
                ]
                if args.num_aioner is not None
                else [0, others_idx[1]]
            )
            tot_label_embedding = (
                tot_label_embedding[dataset_range[0] : dataset_range[1]]
                if len(args.task_list) > 1
                else tot_label_embedding
            )
        args.logger.info(tot_label_embedding)
        args.logger.info(tot_instance_num)
        torch.save(
            tot_label_embedding,
            os.path.join(args.save_label_embedding, "label_embedding.pt"),
        )
