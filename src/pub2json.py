import json
import re
from tabnanny import check
import numpy as np
import argparse
import os

'''train_out = './../bc5cdr/json/train.json'
dev_out = './../bc5cdr/json/dev.json'
test_out = './../bc5cdr/json/test.json'
bc5cdr_train = "../bc5cdr/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt"
bc5cdr_dev = "../bc5cdr/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt"
bc5cdr_test = "../bc5cdr/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"'''


def pub2json(pubtator_file):
    """
    入力：パス，出力：辞書入りリスト
    """
    output_list = []
    output_dict = {}
    sentences = []
    sentence_starts = []
    ne = []
    check_point = False

    with open(pubtator_file, mode="r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            if re.match("^\d+\|t\|", line) is not None:
                length_doc = []
                word_len = []
                sent_len_words = []
                output_dict = {}
                sentences = []
                ne_doc = []
                check_point = True
                doc_num, line = re.split("\|t\|", line)
                length_doc, sentences, sent_len_words, word_len = separate_sentences(
                    line, length_doc, sentences, sent_len_words, word_len
                )
                assert sum(length_doc) == sum(
                    word_len
                ), f"{doc_num} {sum(length_doc)} {sum(word_len)}"
                output_dict["doc_key"] = doc_num
            elif re.match("^\d+\|a\|", line) is not None:
                line = re.split("\|a\|", line)[1]
                assert type(line) == str
                length_doc, sentences, sent_len_words, word_len = separate_sentences(
                    line, length_doc, sentences, sent_len_words, word_len
                )
                output_dict["sentences"] = sentences
                word_starts = len2start(word_len)
                sentence_starts = len2start(length_doc)
                ne_doc = [[] for _ in range(len(sentence_starts))]
            elif re.match("^.+\t\d", line) is not None:
                line = line.split("\t")
                assert len(line) >= 6, line
                start_ne, end_ne, words_list, ne = (
                    int(line[1]),
                    int(line[2]),
                    line[3].split(),
                    line[4],
                )
                idx_sent = search_position(start_ne, sentence_starts)
                idx_word_start = search_position(start_ne, word_starts)
                idx_word_end = len(words_list) + idx_word_start - 1
                ne_taple = (idx_word_start, idx_word_end, ne)

                ne_doc[idx_sent].append(ne_taple)
            else:
                if check_point:
                    output_dict["ner"] = ne_doc
                    output_list.append(output_dict)
                    check_point = False
        return output_list


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    train_out = os.path.join(args.out_dir, "train.json")
    if os.path.isdir(args.data_dir):
        dev_out = os.path.join(args.out_dir, "dev.json")
        test_out = os.path.join(args.out_dir, "test.json")
        data_train = os.path.join(args.data_dir, "train.txt")
        data_dev = os.path.join(args.data_dir, "dev.txt")
        data_test = os.path.join(args.data_dir, "test.txt")
        train_dict = pub2json(data_train)
        dev_dict = pub2json(data_dev)
        test_dict = pub2json(data_test)
        write_file(train_dict, train_out)
        write_file(dev_dict, dev_out)
        write_file(test_dict, test_out)
    else:
        train_dict = pub2json(args.data_dir)
        write_file(train_dict, train_out)
