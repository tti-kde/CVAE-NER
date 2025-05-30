import json
import re

# from tabnanny import check
import numpy as np
import argparse
import os

train_out = "./../corpus/bc4/json/train.json"
bc4_train_text = "/home/user/workspace/data/corpus/bc4"


def pub2json(pubtator_txt: str, pubtator_ano: str) -> list:
    """
    入力：パス，出力：辞書入りリスト
    """
    output_list = []
    output_dict = {}
    sentences = []
    sentence_starts = []
    ne = []

    with open(pubtator_ano, mode="r") as ano_stream:
        annotations = dict(annotations_iter(ano_stream))

    with open(pubtator_txt, mode="r") as txt_stream:
        for pmid, title, abstract in txt_iter(txt_stream):
            title_anns, abstract_anns = annotations[pmid] if pmid in annotations else ([], [])
            # print('doc start')
            length_doc = []
            word_len = []
            sent_len_words = []
            output_dict = {}
            output_dict["doc_key"] = pmid
            sentences = []
            ne_doc = []
            title_len = len(title)
            length_doc, sentences, sent_len_words, word_len = separate_sentences(
                title + " " + abstract, length_doc, sentences, sent_len_words, word_len
            )
            output_dict["sentences"] = sentences
            word_starts = len2start(word_len)
            sentence_starts = len2start(length_doc)
            ne_doc = [[] for _ in range(len(sentence_starts))]

            for _, start_ne, end_ne, ne in title_anns:
                idx_sent = search_position(start_ne, sentence_starts)
                idx_word_start = search_position(start_ne, word_starts)
                idx_word_end = len(ne.split()) + idx_word_start - 1
                ne_tuple = (idx_word_start, idx_word_end, "Chemical")
                ne_doc[idx_sent].append(ne_tuple)
            for _, start_ne, end_ne, ne in abstract_anns:
                start_ne += title_len
                idx_sent = search_position(start_ne, sentence_starts)
                idx_word_start = search_position(start_ne, word_starts)
                idx_word_end = len(ne.split()) + idx_word_start - 1
                ne_list = [idx_word_start, idx_word_end, "Chemical"]
                ne_doc[idx_sent].append(ne_list)
            output_dict["ner"] = ne_doc
            output_list.append(output_dict)
    return output_list


def txt_iter(stream):
    for line in stream:
        yield line.strip().split("\t")


def annotations_iter(stream):
    current_pmid, title_anns, abstract_anns = None, None, None
    lno = 0

    try:
        for lno, line in enumerate(stream, 1):
            pmid, section, start, end, text, infon = line.strip().split("\t")
            ann = (infon, int(start), int(end), text)

            if pmid != current_pmid:
                if current_pmid is not None:
                    yield current_pmid, (title_anns, abstract_anns)

                current_pmid, title_anns, abstract_anns = pmid, [], []

            if section == "A":
                abstract_anns.append(ann)
            elif section == "T":
                title_anns.append(ann)
            else:
                raise RuntimeError('unknown annotation section "%s"' % section)
    except UnicodeDecodeError as e:
        raise RuntimeError("encoding error in %s at line %d: %s" % (stream.name, lno, str(e)))


def len2start(len_list):
    starts_list = np.cumsum(len_list)  # 累積和
    starts_list = np.roll(starts_list, 1)  # 一つずらす
    starts_list[0] = 0
    return starts_list


def search_position(idx, starts_list):
    """
    開始位置のidxから何番目に含まれるか
    """
    idx_max = np.maximum(starts_list, idx).tolist()
    position_idx = idx_max.count(idx) - 1
    return position_idx


def separate_sentences(doc, length_sent_sum, sent_list, length_words_in_sent, length_words_in_doc):
    doc_sent = re.sub("\.\s([A-Z])", "\t\\1", doc)
    doc_list = [sentence for sentence in re.split("(\t)|(\.$)", doc_sent) if sentence is not None]
    for sentence in doc_list:
        if sentence == "." or sentence == "" or sentence == "\t":
            continue
        length_sent_sum.append(len(sentence) + 2)
        sentence = sentence.split()
        sentence_list = sentence + ["."]
        sent_list.append(sentence_list)
        word_len_sent = [len(word) + 1 for word in sentence]
        word_len_sent[-1] -= 1
        word_len_sent.append(2)
        length_words_in_doc += word_len_sent
        length_words_in_sent.append(len(sentence_list))
    return length_sent_sum, sent_list, length_words_in_sent, length_words_in_doc


def write_file(dict, out_dir):
    with open(out_dir, "w") as f:
        for data in dict:
            f.write("{}\n".format(json.dumps(data)))


if __name__ == "__main__":
    """train_dict = pub2json(bc4_train)
    dev_dict = pub2json(bc4_dev)
    test_dict = pub2json(bc4_test)
    write_file(train_dict, train_out)
    write_file(dev_dict, dev_out)
    write_file(test_dict, test_out)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--data_type", type=str, choices=["training", "development", "evaluation"])
    args = parser.parse_args()

    if args.data_type == "training":
        out_name = "train.json"
    elif args.data_type == "development":
        out_name = "dev.json"
    else:
        out_name = "test.json"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    data_out = os.path.join(args.out_dir, out_name)
    data_txt = os.path.join(args.data_dir, f"{args.data_type}.abstracts.txt")
    data_ano = os.path.join(args.data_dir, f"{args.data_type}.annotations.txt")
    data_dict = pub2json(data_txt, data_ano)
    write_file(data_dict, data_out)
