import re
from tabnanny import check
import numpy as np

train_out = "./../bc5cdr/json/train.json"
dev_out = "./../bc5cdr/json/dev.json"
test_out = "./../bc5cdr/json/test.json"
bc5cdr_train = "../bc5cdr/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt"
bc5cdr_dev = "../bc5cdr/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt"
bc5cdr_test = "../bc5cdr/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"


def pub2brat(pubtator_file):
    """
    入力：パス，出力：辞書入りリスト
    """
    doc_list = []
    ne_list = []
    re_list = []
    ne_num = 0
    re_num = 0

    with open(pubtator_file, mode="r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            if re.match("^\d+\|t\|", line) is not None:
                # print('doc start')
                doc = re.split("\|t\|", line)[1]
            elif re.match("^\d+\|a\|", line) is not None:
                line = re.split("\|a\|", line)[1]
                doc += line
                doc_list.append(doc)
            elif re.match("^.+\t\d", line) is not None:
                ne_num += 1
                line = line.split("\t")
                assert len(line) >= 6, line
                start_ne, end_ne, ne, word = int(line[1]), int(line[2]), line[4], line[5]
                ne = "T{}\t{}\s{}\s{}\t{}".format(ne_num, ne, start_ne, end_ne, word)
                ne_list.append(ne)
            else:
                line = line.split("\t")
                re, words = line[1], line[2]
                words = words.split()
                ne = "R{}\t{}\s{}\s{}".format(re_num, re, word[0], word[2])
        return doc_list, ne_list, re_list


if __name__ == "__main__":
    train_doc, train_ne, train_re = pub2brat(bc5cdr_train)
    dev_doc, dev_ne, dev_re = pub2brat(bc5cdr_dev)
    test_doc, test_ne, test_re = pub2brat(bc5cdr_test)
