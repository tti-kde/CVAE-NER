import os
from typing import List
from shared.const import task_ner_labels, general_label_map_other
import torch

from label_model import NERModel
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    T5EncoderModel,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model  # , TaskType
from transformers import AdamW


def initialize_optimizer(args: dict, model: object) -> object:
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if "bias" in n and "layer_norm" in n],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "encoder" in n and "bias" not in n and "layer_norm" not in n
            ]
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "encoder" not in n and "label_embedding" in n and "bias" not in n
            ],
            "lr": args.learning_rate * 0.5,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "encoder" not in n and "bias" not in n and "label_embedding" not in n
            ],
            "lr": args.task_learning_rate,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
    )
    return optimizer


def initialize_model(
    args: dict,
    num_ner_labels: int,
    mask_range=None,
) -> object:
    model_name = args.model

    if mask_range[0] is not None:
        corpus_labels_mask = make_labels_mask(args, num_ner_labels, mask_range)
    else:
        corpus_labels_mask = None
    if args.do_train or args.lookup_label_token_onehot == "only_shared":
        one_hot_label = make_one_hot_label(args, num_ner_labels, mask_range)
    else:
        one_hot_label = torch.zeros(num_ner_labels, num_ner_labels + mask_range[0][1]).to(
            args.model_device
        )
    config = AutoConfig.from_pretrained(args.model)
    args.hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NERModel(
        config,
        num_ner_labels=num_ner_labels,
        max_span_length=args.max_span_length,
        mask_range=mask_range,
        args=args,
        one_hot_label=one_hot_label,
    )

    # use T5 encoder
    if args.use_t5:
        if args.bool_quantize:
            model.encoder, tokenizer, config = quantize_model(model_name, tokenizer, args, config)
        else:
            model.encoder = T5EncoderModel.from_pretrained(model_name)

        if args.use_lora:
            model.encoder = get_lora_model(args, model.encoder)
    else:
        model.encoder = AutoModel.from_pretrained(args.model)
    if args.bert_model_dir is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(args.bert_model_dir, "model_weights.bin"),
                map_location=args.model_device,
            )
        )

    if args.general_one_hot:
        args.logger.info("Using general one hot")
        args.logger.info(f"general_one_hot: {general_label_map_other}")
    return args, model, tokenizer, corpus_labels_mask


def make_labels_mask(args: dict, num_ner_labels: int, mask_range: list) -> dict:
    corpus_labels_mask = {}
    for r, corpus in zip(mask_range, args.task_list):
        labels_mask = torch.zeros(1, num_ner_labels)
        if args.not_corpus_difference and r[-1] != 7:
            labels_mask[0] = 1
            for i in r:
                labels_mask[:, i] = 1
        else:
            labels_mask[:, r[0] : r[1]] = 1
        corpus_labels_mask[corpus] = labels_mask
    return corpus_labels_mask


def add_special_tokens(tokenizer, args, config):
    if args.corpus_token_pos:
        corpus_special_tokens = [f"[{corpus}]" for corpus in args.task_list]
        tokenizer.add_special_tokens({"additional_special_tokens": corpus_special_tokens})
        config.vocab_size += len(corpus_special_tokens)
    if args.label_token:
        label_special_tokens = [f"[other_{corpus}]" for corpus in args.task_list]
        label_special_tokens += [
            f"[{label}_{corpus}]" for corpus in args.task_list for label in task_ner_labels[corpus]
        ]
        tokenizer.add_special_tokens({"additional_special_tokens": label_special_tokens})
        config.vocab_size += len(label_special_tokens)
    elif args.shared_label_token:
        label_special_tokens = [f"[other_{corpus}]" for corpus in args.task_list]
        label_special_tokens += [f"[{label}]" for label in task_ner_labels["bc8"]]
        for corpus in args.task_list[1:]:
            for i, label in enumerate(task_ner_labels[corpus]):
                if i + 1 in general_label_map_other[corpus].keys():
                    continue
                label_special_tokens.append(f"[{label}]")
        tokenizer.add_special_tokens({"additional_special_tokens": label_special_tokens})
        config.vocab_size += len(label_special_tokens)
    return tokenizer, config


def quantize_model(model_name, tokenizer, args, config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit 量子化を有効化します。
        bnb_4bit_use_double_quant=True,  # ネストされた量子化 (Nested quantization) を有効化します。
        bnb_4bit_quant_type="nf4",  # 量子化データタイプを設定します。nf4 は 4-bit NormalFloat Quantization のデータ型です。
        bnb_4bit_compute_dtype=torch.float32,  # torch.bfloat16,  # 量子化計算時のデータタイプを設定します。
    )
    encoder = T5EncoderModel.from_pretrained(
        model_name,
        device_map={"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))},
        quantization_config=bnb_config,
    )
    tokenizer, config = add_special_tokens(tokenizer, args, config)
    if args.corpus_token_pos or args.label_token or args.shared_label_token:
        encoder.resize_token_embeddings(config.vocab_size)
    for param in encoder.parameters():
        # ベースモデルのパラメータは勾配計算の対象外とします。
        param.requires_grad = False
        # 学習に使用するレイヤーのパラメータの型を float32 にして精度を上げます。
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # メモリを節約するために、Gradient Checkpointing アルゴリズムを有効化します。
    encoder.gradient_checkpointing_enable()

    # モデルの重みを固定したままアダプターの重みを調整するために、入力埋め込みグラデーションを有効化します。
    encoder.enable_input_require_grads()
    return encoder, tokenizer, config


def get_lora_model(args, encoder):
    target_modules = ["q", ".k", "v", "o", "wi", "wo"]
    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    encoder = get_peft_model(encoder, lora_config)
    print_trainable_parameters(args, encoder)
    return encoder


def pad_spans_tensors(samples: List, tokenizer: object) -> List:
    output_list = len(samples) * [None]
    tokens_tensor_list = []
    tokens_len_list = []
    bert_spans_tensor_list = []
    spans_ner_label_tensor_list = []
    spans_word_list = []
    word_id2offset_list = []

    max_tokens = 0
    max_tokens_len = 0
    max_spans = 0
    max_sentences = 0
    for i, sample in enumerate(samples):
        tokens_tensor = sample["tokens"]
        tokens_len = np.array(sample["tokens_len"])
        bert_spans_tensor = sample["spans"]
        spans_ner_label_tensor = sample["spans_label"]
        word_id2offset = np.array(sample["word_id2offset"])
        if spans_ner_label_tensor[0] == 7:
            assert spans_ner_label_tensor[-1] != 0
        tokens_tensor_list.append(tokens_tensor)
        tokens_len_list.append(tokens_len)
        bert_spans_tensor_list.append(bert_spans_tensor)
        spans_ner_label_tensor_list.append(spans_ner_label_tensor)
        spans_word_list.append(np.array(sample["spans_word"]))
        word_id2offset_list.append(word_id2offset)
        assert bert_spans_tensor.shape[-2] == spans_ner_label_tensor.shape[-1], (
            bert_spans_tensor.shape,
            spans_ner_label_tensor.shape,
        )
        if tokens_tensor.shape[-1] > max_tokens:
            max_tokens = tokens_tensor.shape[-1]
        if tokens_len.shape[0] > max_tokens_len:
            max_tokens_len = len(tokens_len)
        if bert_spans_tensor.shape[-2] > max_spans:
            max_spans = bert_spans_tensor.shape[-2]
        if len(word_id2offset) > max_sentences:
            max_sentences = word_id2offset.shape[0]

    # apply padding and concatenate tensors
    for i, (
        sample,
        tokens_tensor,
        tokens_len,
        bert_spans_tensor,
        spans_ner_label_tensor,
        word_id2offset,
        spans,
    ) in enumerate(
        zip(
            samples,
            tokens_tensor_list,
            tokens_len_list,
            bert_spans_tensor_list,
            spans_ner_label_tensor_list,
            word_id2offset_list,
            spans_word_list,
        )
    ):
        output_sample = sample.copy()
        # padding for tokens
        num_tokens = tokens_tensor.shape[-1]
        tokens_pad_length = max_tokens - num_tokens
        attention_tensor = torch.full([num_tokens], 1, dtype=torch.long)
        if tokens_pad_length > 0:
            pad = torch.full(
                [tokens_pad_length],
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            tokens_tensor = torch.cat((tokens_tensor, pad), dim=0)
            attention_pad = torch.full([tokens_pad_length], 0, dtype=torch.long)
            attention_tensor = torch.cat((attention_tensor, attention_pad), dim=0)
        # padding for tokens_len
        num_tokens_len = tokens_len.shape[0]
        tokens_len_pad_length = max_tokens_len - num_tokens_len
        if tokens_len_pad_length > 0:
            pad = np.full((tokens_len_pad_length), 0)
            tokens_len = np.concatenate((tokens_len, pad), axis=0)

        # padding for spans
        num_spans = bert_spans_tensor.shape[-2]
        spans_pad_length = max_spans - num_spans
        spans_mask_tensor = torch.full([num_spans], 1, dtype=torch.long)
        assert spans_ner_label_tensor.size() == spans_mask_tensor.size()
        if spans_pad_length > 0:
            pad = torch.full(
                [spans_pad_length, bert_spans_tensor.shape[-1]],
                0,
                dtype=torch.long,
            )
            bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=0)
            mask_pad = torch.full([spans_pad_length], 0, dtype=torch.long)
            spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=0)
            spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=0)
            # padding for spans_word
            pad = np.full((spans_pad_length, spans.shape[-1]), 0)
            spans = np.concatenate((spans, pad), axis=0)

        # padding for word_id2offset
        num_sentences = word_id2offset.shape[0]
        sentences_pad_length = max_sentences - num_sentences
        if sentences_pad_length > 0:
            pad = np.full((sentences_pad_length), 0)
            word_id2offset = np.concatenate((word_id2offset, pad), axis=0)
        output_sample["tokens"] = tokens_tensor
        output_sample["tokens_len"] = tokens_len
        output_sample["attention_mask"] = attention_tensor
        output_sample["spans"] = bert_spans_tensor
        output_sample["spans_mask"] = spans_mask_tensor
        output_sample["spans_label"] = spans_ner_label_tensor
        output_sample["spans_word"] = spans
        output_sample["word_id2offset"] = word_id2offset
        output_list[i] = output_sample
    return output_list


def make_label_embedding(args, model, batch, num_ner_labels, corpus_labels_mask):
    (
        tokens_tensor,
        attention_mask_tensor,
        bert_spans_tensor,
        spans_mask_tensor,
        spans_ner_label_tensor,
    ) = (
        batch["tokens"],
        batch["attention_mask"],
        batch["spans"],
        batch["spans_mask"],
        batch["spans_label"],
    )
    if len(args.task_list) > 1:
        labels_mask = (
            corpus_labels_mask[batch["corpus_name"][0]]
            if not args.not_corpus_difference
            else torch.ones(1, num_ner_labels)
        )
    else:
        labels_mask = torch.ones(1, num_ner_labels)
    with torch.autocast("cuda", dtype=torch.float32):
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=tokens_tensor.to(args.model_device),
                spans=bert_spans_tensor.to(args.model_device),
                spans_mask=spans_mask_tensor.to(args.model_device),
                spans_ner_label=spans_ner_label_tensor.to(args.model_device),
                attention_mask=attention_mask_tensor.to(args.model_device),
                labels_mask=labels_mask.to(args.model_device),
                training=False,
            )
            num_labels = output[0].size(-1)
            spans_embedding = output[1][spans_mask_tensor == 1]
            spans_ner_label_tensor = spans_ner_label_tensor[spans_mask_tensor == 1]
            label_embedding = torch.zeros(
                num_labels, spans_embedding.size(-1), device=args.model_device
            )
            num_instances = torch.zeros(num_labels, device=args.model_device)
            for i in range(num_labels):
                label_embedding[i] = torch.sum(spans_embedding[spans_ner_label_tensor == i], 0)
                num_instances[i] = torch.where(spans_ner_label_tensor == i, 1, 0).sum()
    return label_embedding, num_instances


def make_one_hot_label(args, num_ner_labels, mask_range):
    one_hot_label = torch.eye(num_ner_labels)
    if (
        args.general_one_hot or args.lookup_label_token_onehot
    ) and args.lookup_label_token_onehot != "separate":
        general_ner_label = list(range(mask_range[0][0], mask_range[0][1]))
        for corpus_num, (corpus, corpus_range) in enumerate(
            zip(args.task_list[: args.num_aioner + 1], mask_range[: args.num_aioner + 1])
        ):
            if corpus_num == 0:
                continue
            for label_index in range(corpus_range[0], corpus_range[1]):
                if label_index - corpus_range[0] in general_label_map_other[corpus].keys():
                    general_ner_label += [
                        general_label_map_other[corpus][label_index - corpus_range[0]]
                    ]
                else:
                    general_ner_label += [mask_range[0][1]]
        general_ner_label = torch.eye(mask_range[0][1] + 1)[general_ner_label]
        general_ner_label = general_ner_label[:, : mask_range[0][1]]
        assert general_ner_label.size(1) == mask_range[0][1]
        if args.lookup_label_token_onehot == "only_shared":
            one_hot_label = general_ner_label.to(args.model_device)
        else:
            one_hot_label = torch.cat((one_hot_label, general_ner_label), dim=1).to(
                args.model_device
            )
    return one_hot_label


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    args.logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def run_batch(
    batch: list,
    args: dict,
    model: NERModel,
    num_ner_labels: int,
    corpus_labels_mask: dict,
    mask_range: list,
    training: bool = True,
    accelerator: object = None,
):

    # convert samples to input tensors
    (
        tokens_tensor,
        attention_mask_tensor,
        bert_spans_tensor,
        spans_mask_tensor,
        spans_ner_label_tensor,
    ) = (
        batch["tokens"],
        batch["attention_mask"],
        batch["spans"],
        batch["spans_mask"],
        batch["spans_label"],
    )
    if len(args.task_list) > 1:
        if args.dataset_batch_size is None or not training:
            labels_mask = (
                corpus_labels_mask[batch["corpus_name"][0]]
                if not args.not_corpus_difference
                else torch.ones(1, num_ner_labels)
            )
        else:
            labels_mask = torch.zeros(
                args.train_batch_size, bert_spans_tensor.size(1), num_ner_labels
            )
            for n in range(args.train_batch_size // args.dataset_batch_size):
                corpus_name = batch["corpus_name"][n * args.dataset_batch_size]
                labels_mask[
                    n * args.dataset_batch_size : (n + 1) * args.dataset_batch_size,
                    :,
                    :,
                ] = corpus_labels_mask[corpus_name]
    else:
        labels_mask = torch.ones(1, num_ner_labels)
    output_dict = {
        "ner_loss": 0,
    }

    with torch.autocast("cuda", dtype=torch.float32):  # torch.bfloat16
        if training:
            model.train()
            output = model(
                input_ids=tokens_tensor.to(args.model_device),
                spans=bert_spans_tensor.to(args.model_device),
                spans_mask=spans_mask_tensor.to(args.model_device),
                spans_ner_label=spans_ner_label_tensor.to(args.model_device),
                attention_mask=attention_mask_tensor.to(args.model_device),
                labels_mask=labels_mask.to(args.model_device),
            )
            output_dict["ner_loss"] = output[0].mean()
            if args.entity_const:
                output_dict["entity_const_loss"] = output[-1].mean()
            if not args.simple:
                output_dict["class_loss"] = output[-3].mean()
                output_dict["cvae_loss"] = output[-2].mean()
        else:
            model.eval()
            with torch.no_grad():
                output = model(
                    input_ids=tokens_tensor.to(args.model_device),
                    spans=bert_spans_tensor.to(args.model_device),
                    spans_mask=spans_mask_tensor.to(args.model_device),
                    spans_ner_label=spans_ner_label_tensor.to(args.model_device),
                    attention_mask=attention_mask_tensor.to(args.model_device),
                    labels_mask=labels_mask.to(args.model_device),
                    training=False,
                )

            _, predicted_label = output[0].max(2)

            predicted_label = predicted_label[spans_mask_tensor == 1]
            target_label = spans_ner_label_tensor[spans_mask_tensor == 1]
            spans_embedding = output[1][spans_mask_tensor == 1]
            predicted_label = accelerator.pad_across_processes(predicted_label, pad_index=-100)
            target_label = accelerator.pad_across_processes(target_label, pad_index=-100)
            spans_embedding = accelerator.pad_across_processes(spans_embedding, pad_index=-100)
            predicted_label = accelerator.gather(predicted_label)
            target_label = accelerator.gather(target_label)
            spans_embedding = accelerator.gather(spans_embedding)

            output_dict["pred_ner"] = predicted_label.cpu().numpy()
            output_dict["target_ner"] = target_label.cpu().numpy()
            if args.type_tsne:
                output_dict["spans_embedding"] = spans_embedding.detach().clone()

            if args.type_tsne:
                spans_ner_label_tensor = accelerator.gather_for_metrics((spans_ner_label_tensor))
                output_dict["label_tensor"] = spans_ner_label_tensor.detach().clone().cpu().numpy()
    return output_dict
