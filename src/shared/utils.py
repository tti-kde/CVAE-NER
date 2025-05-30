import numpy as np
import json
import torch
from shared.const import task_ner_labels, general_label_map_other, aioner_original


def get_input_tensors(args, tokenizer, tokens, spans, spans_ner_label, corpus_name):
    start2idx = []
    end2idx = []
    eos_token = tokenizer.eos_token if args.use_t5 else tokenizer.sep_token

    bert_tokens = []
    if not args.use_t5:
        bert_tokens.append(tokenizer.cls_token)
    if args.corpus_token_pos == "start":
        bert_tokens.append(f"[{corpus_name}]")
    for token in tokens:
        start2idx.append(len(bert_tokens))
        sub_tokens = tokenizer.tokenize(token)
        bert_tokens += sub_tokens
        end2idx.append(len(bert_tokens) - 1)
    bert_tokens.append(eos_token)
    if args.corpus_token_pos == "end":
        bert_tokens.append(f"[{corpus_name}]")
    if args.label_token:
        bert_tokens.append(f"[other_{corpus_name}]")
        for label in task_ner_labels[corpus_name]:
            bert_tokens.append(f"[{label}_{corpus_name}]")
    if args.shared_label_token:
        bert_tokens.append(f"[other_{corpus_name}]")
        general_dict = general_label_map_other[corpus_name] if corpus_name != "bc8" else {}
        for i, label in enumerate(task_ner_labels[corpus_name]):
            if i + 1 in general_dict.keys():
                general_idx = general_dict[i + 1]
                label = task_ner_labels["bc8"][general_idx - 1]
            bert_tokens.append(f"[{label}]")
    bert_tokens.append(eos_token)
    if len(bert_tokens) > 512:
        args.logger.info("Too long to encode: %d" % len(bert_tokens))

    indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
    tokens_tensor = torch.tensor(indexed_tokens)

    bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
    bert_spans_tensor = torch.tensor(bert_spans)

    spans_ner_label_tensor = torch.tensor(spans_ner_label)

    return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor


def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False


def convert_dataset_to_samples(
    dataset,
    tokenizer,
    ner_label2id=None,
    data_num=None,
    others_idx=None,
    labels=None,
    disease_limit=None,
    args=None,
    bool_chebi=False,
):
    """
    Extract sentences and gold entities from a dataset
    """
    samples = []
    num_ners = {}
    if labels is not None:
        for label in labels:
            num_ners[label] = 0
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0
    max_ne_len = 0
    num_long_ne = 0
    num_ne = 0
    num_spans = 0

    if args.split == 0:
        data_range = (0, len(dataset))
    elif args.split == 1:
        data_range = (int(len(dataset) * 0.8), len(dataset))
    elif args.split == 2:
        data_range = (0, int(len(dataset) * 0.8))
    elif args.split == 100:
        data_range = (aioner_original[args.task_list[0]], len(dataset))
        args.split += 1
    elif args.split == 101:
        data_range = (0, aioner_original[args.task_list[0]])

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            if labels is not None:
                for ner in sent.ner:
                    num_ners[ner.label] += 1
            num_ner += len(sent.ner)
            sample = {
                "doc_key": doc._doc_key,
                "sentence_ix": sent.sentence_ix,
            }
            if args.context_window != 0 and len(sent.text) > args.context_window:
                if bool_chebi:
                    continue
                args.logger.info("Long sentence: {} {}".format(sample, len(sent.text)))
            tokens = sent.text
            sample["sent_length"] = len(sent.text)
            sample["word_id2offset"] = sent.word_id2offset
            sent_start = 0
            sent_end = len(tokens)

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))

            original_tokens = tokens
            if args.context_window > 0:
                add_left = (args.context_window - len(sent.text)) // 2
                add_right = (args.context_window - len(sent.text)) - add_left

                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            sample["sent_start"] = sent_start
            sample["sent_end"] = sent_end
            sample["sent_start_in_doc"] = sent.sentence_start
            sent_ner = {}
            for ner in sent.ner:
                sent_ner[ner.span.span_sent] = ner.label
                max_ne_len = max(max_ne_len, len(ner.span.text))
                if len(ner.span.text) > args.max_span_length:
                    num_long_ne += 1

            span2id = {}
            spans = []
            spans_ner_label = []
            if len(args.task_list) > 1:
                sample["corpus_name"] = args.task_list[data_num if data_num is not None else 0]
            else:
                sample["corpus_name"] = args.task_list[0]
            for i in range(len(sent.text)):
                for j in range(i, min(len(sent.text), i + args.max_span_length)):
                    spans.append((i + sent_start, j + sent_start, j - i + 1))
                    span2id[(i, j)] = len(spans) - 1
                    if args.special_ne is not None:
                        if " ".join(original_tokens[i : j + 1]) in args.special_ne:
                            spans_ner_label.append(
                                100 + args.special_ne.index(" ".join(original_tokens[i : j + 1]))
                            )
                            if (i, j) in sent_ner:
                                num_ne += 1
                        elif (i, j) not in sent_ner:
                            if data_num is None:
                                spans_ner_label.append(0)
                            else:
                                spans_ner_label.append(others_idx[data_num])
                        else:
                            if data_num is None:
                                spans_ner_label.append(ner_label2id[sent_ner[(i, j)]])
                            else:
                                spans_ner_label.append(
                                    ner_label2id[f"{sent_ner[(i, j)]}:{data_num}"]
                                )
                    elif (i, j) not in sent_ner:
                        if data_num is None or args.not_corpus_difference:
                            spans_ner_label.append(0)
                        else:
                            spans_ner_label.append(others_idx[data_num])
                    elif (
                        sent_ner[(i, j)] not in ner_label2id.keys()
                        and f"{sent_ner[(i, j)]}:{data_num}" not in ner_label2id.keys()
                        and args.case_tsne is None
                    ):
                        if not args.not_corpus_difference:
                            args.logger.info("Wrong NER label: %s" % sent_ner[(i, j)])
                        if data_num is None or args.not_corpus_difference:
                            spans_ner_label.append(0)
                        else:
                            spans_ner_label.append(others_idx[data_num])
                    else:
                        num_ne += 1
                        if data_num is None:
                            spans_ner_label.append(ner_label2id[sent_ner[(i, j)]])  # dummy label: 0
                        elif sent_ner[(i, j)] == 500:
                            spans_ner_label.append(500)
                        else:
                            spans_ner_label.append(ner_label2id[f"{sent_ner[(i, j)]}:{data_num}"])
            (
                tokens_tensor,
                bert_spans_tensor,
                spans_ner_label_tensor,
            ) = get_input_tensors(
                args, tokenizer, tokens, spans, spans_ner_label, sample["corpus_name"]
            )
            sample["tokens"] = tokens_tensor
            sample["tokens_len"] = [len(token) for token in tokens]
            sample["spans"] = bert_spans_tensor
            sample["spans_word"] = spans
            sample["spans_label"] = spans_ner_label_tensor
            num_spans += len(spans)
            samples.append(sample)
            if disease_limit is not None:
                if disease_limit < num_ners[labels[0]]:
                    break
    args.logger.info("Number of ne: %d" % num_ne)
    avg_length = sum([sample["tokens"].size(0) for sample in samples]) / len(samples)
    max_length = max([sample["tokens"].size(0) for sample in samples])
    args.logger.info("# Overlap: %d" % num_overlap)
    args.logger.info(
        "Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length"
        % (len(samples), data_range[1] - data_range[0], num_ner, avg_length, max_length)
    )
    args.logger.info(
        "Max Length: %d, max NER_in_sent: %d, max_len_NER: %d, long NER: %d"
        % (max_len, max_ner, max_ne_len, num_long_ne)
    )
    return samples, num_ner, num_ners, num_spans


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_train_fold(data, fold):
    print("Getting train fold %d..." % fold)
    len_data = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < len_data or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print("# documents: %d --> %d" % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def get_test_fold(data, fold):
    print("Getting test fold %d..." % fold)
    len_data = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= len_data and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print("# documents: %d --> %d" % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    range_vector = [i for i in range(size)]
    if device > -1:
        return torch.tensor(range_vector, dtype=torch.long, device=f"cuda:{device}")
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(
    indices: torch.LongTensor, sequence_length: int
) -> torch.LongTensor:
    # Shape: (batch_size)
    assert torch.max(indices) < sequence_length, f"{torch.max(indices)} {sequence_length}"
    assert torch.min(indices) >= 0, f"{torch.min(indices)}"
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
) -> torch.Tensor:
    # Shape: (batch_size * d_1 * ... * d_n)
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = torch.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L
