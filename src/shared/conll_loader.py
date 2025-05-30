import re
import json

from torch import le, long

doc_name = [
    "BC5_chem",
    "BC5_dis",
    "BioID",
    "NLM_chem",
    "GNormPlus",
    "Linnaeus",
    "ncbi",
    "NLMgene",
    "Species_800",
    "tmVar3",
]


def conll2json(file):
    """Converts a CoNLL file to a JSON file."""
    output = []
    with open(file, "r") as f:
        lines = f.readlines()
        sentence = []
        sentences = []
        flag_in_tag = False
        word_count = 0
        dataset_count = 0
        num_long_sentence = 0
        num_long_word = 0
        flag_long_word = False
        ner = []
        ner_in_sentence = []
        current_tag = "<Chemical>"
        for line in lines:
            if line == "\n":
                continue
            word, tag = list(filter(None, re.split(r"\t|\n", line)))
            if re.fullmatch(r"</.*>", word) is not None:
                if len(sentence) > 150:
                    num_long_sentence += 1
                    word_count -= len(sentence)
                elif flag_long_word:
                    word_count -= len(sentence)
                    flag_long_word = False
                else:
                    sentences.append(sentence)
                    ner.append(ner_in_sentence)
                sentence = []
                ner_in_sentence = []
            elif re.fullmatch(r"<.*>", word) is not None:
                if current_tag != word:
                    word_count = 0
                    if current_tag != "<ALL>":
                        output_dict = {
                            "doc_key": f"{dataset_count}",
                            "sentences": sentences,
                            "ner": ner,
                        }
                        with open(
                            f"/workspace/data/corpus/aioner/json/{doc_name[dataset_count]}.json",
                            "w",
                        ) as f:
                            txt = json.dumps(output_dict)
                            f.write(txt)
                        dataset_count += 1
                        output.append(output_dict)
                    ner = []
                    sentences = []
                    current_tag = word
            else:
                sentence.append(word)
                if len(word) > 50:
                    print(word)
                    num_long_word += 1
                    flag_long_word = True
                if flag_in_tag and re.fullmatch(r"O-.*", tag) is not None:
                    flag_in_tag = False
                    ner_in_sentence.append([start, word_count, f"AIO_{current_tag[1:-1]}"])
                elif re.fullmatch(r"B-.*", tag) is not None:
                    flag_in_tag = True
                    start = word_count
                word_count += 1
        if current_tag != "<ALL>":
            output_dict = {
                "doc_key": doc_name[dataset_count],
                "sentences": sentences,
                "ner": ner,
            }
            with open(
                f"/workspace/data/corpus/aioner/json/{doc_name[dataset_count]}.json",
                "w",
            ) as f:
                txt = json.dumps(output_dict)
                f.write(txt)
            dataset_count += 1
            output.append(output_dict)
        print(num_long_sentence)
        print(num_long_word)
    return output


if __name__ == "__main__":
    path = "/workspace/data/corpus/aioner/conll/Merged_All-AIO.conll"
    output = conll2json(path)
    print(len(output))
