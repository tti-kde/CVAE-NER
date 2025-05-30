import json
import re
from datasets import load_dataset
from pyparsing import traceParseAction
from shared.bc8_data_reshape import (
    len2start,
    search_position,
    separate_sentences,
    separate_on_signs,
)


def bigbio2input(js: list) -> list:
    """
    入力：パス，出力：辞書入りリスト
    """
    output_list = []

    for document in js:
        doc_sentences = ""
        for sentence in document["passages"]:
            doc_sentences += sentence["text"][0]
        if re.fullmatch(r"\s*\n*\s*", doc_sentences) is not None:
            continue
        length_doc = []
        word_len = []
        sent_len_words = []
        output_dict = {}
        output_dict["doc_key"] = document["document_id"]
        output_dict["sentences"] = []
        sentences = []
        ne_doc = []
        word_id2offset_doc = []
        length_doc, sentences, sent_len_words, word_len, _ = separate_sentences(
            doc_sentences, length_doc, sentences, sent_len_words, word_len, word_id2offset_doc
        )
        output_dict["sentences"] = sentences
        assert len(word_len) != 0, doc_sentences
        word_starts = len2start(word_len)
        sentence_starts = len2start(length_doc)
        ne_doc = [[] for _ in range(len(sentence_starts))]

        for ano in document["entities"]:
            idx_sent = search_position(ano["offsets"][0][0], sentence_starts)
            idx_word_start = search_position(ano["offsets"][0][0], word_starts)
            ent = separate_on_signs(ano["text"][0]).split()
            idx_word_end = len(ent) + idx_word_start - 1
            ne_tuple = (idx_word_start, idx_word_end, ano["type"])
            ne_doc[idx_sent].append(ne_tuple)
        output_dict["ner"] = ne_doc
        output_list.append(output_dict)
    return output_list


if __name__ == "__main__":
    f = open("/workspace/src/bigbio.txt")
    bigbio_datasets = f.read().splitlines()
    f.close()
    output_dict = {}
    for i, dataset in enumerate(bigbio_datasets):
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
        if dataset == "progene":
            extra_train_data = load_dataset(
                "bigbio/" + dataset, name=f"{dataset_tag}_bigbio_kb", split="split_0_train"
            )
            ne_type_list = bigbio2input(extra_train_data)
        else:
            extra_train_data = load_dataset("bigbio/" + dataset, name=f"{dataset_tag}_bigbio_kb")
            ne_type_list = bigbio2input(extra_train_data["train"])
        output_dict[f"{dataset}"] = ne_type_list
    with open("/workspace/src/ent_type.txt", "w") as f:
        txt = json.dumps(output_dict)
        f.write(txt)
