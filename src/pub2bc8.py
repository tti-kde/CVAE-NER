from pydoc import doc
import re
import json
import argparse
from shared.bc8_data_reshape import separate_on_signs


def pub2bc8(pubtator_file):
    """
    入力：パス，出力：辞書入りリスト
    """
    out_put_dict = {"source": "pubtator", "documents": []}
    ne_types = {}

    with open(pubtator_file, mode="r") as f:
        for line in f:
            line = line.strip()

            if re.match("^\d+\|t\|", line) is not None:
                document = {}
                title = {"offset": 0}
                title_ne_list = []
                abst_ne_list = []
                ne_num = 0

                line = re.split("\|t\|", line)
                document["id"] = line[0]
                title["text"] = line[1]
                title_off = len(title["text"]) + 1
            elif re.match("^\d+\|a\|", line) is not None:
                line = re.split("\|a\|", line)[1]
                line = re.sub("(et al|e\.g|i\.e|cf)\.", "\\1:", line)
                doc_sent = re.sub("(\.)\s([A-Z])", "\\1\t\\2", line)
                line_sent = re.split("\t", doc_sent)
                doc_sent = separate_on_signs(doc_sent)
                doc_sent = separate_on_signs(doc_sent)
                doc_sent = re.split("\t", doc_sent)
                removed_line = ""
                removed_sent_info = {}
                # remove long word
                for sent, sent_origin in zip(doc_sent, line_sent):
                    sent_word = re.split("\s", sent)
                    if len(sent_word) > 200:
                        removed_sent_info[title_off + len(removed_line)] = sent + " "
                        print(document["id"])
                        print(sent)
                        continue
                    bool_long_word = False
                    for word in sent_word:
                        if len(word) > 50:
                            removed_sent_info[title_off + len(removed_line)] = sent + " "
                            bool_long_word = True
                            print(document["id"])
                            print(word)
                            break
                    if not bool_long_word:
                        removed_line += sent_origin + " "
                abst = {"offset": title_off, "text": removed_line[:-1]}
            elif re.match("^.+\t\d", line) is not None:
                ann = {"id": ne_num}
                ne_num += 1
                line = line.split("\t")
                if len(line) >= 6:
                    _, start_ne, end_ne, text, ne_type, ne_id = line
                else:
                    _, start_ne, end_ne, text, ne_type = line
                    ne_id = "-"
                bool_removed_ne = False
                for remove_off in removed_sent_info.keys():
                    if int(start_ne) >= remove_off:
                        start_ne = int(start_ne) - len(removed_sent_info[remove_off])
                        end_ne = int(end_ne) - len(removed_sent_info[remove_off])
                        if start_ne < remove_off:
                            bool_removed_ne = True
                            break
                if not bool_removed_ne:
                    ne_types[ne_type] = 0
                    ann["infons"] = {"identifier": ne_id, "type": ne_type}
                    if (
                        document["id"] == "7520377"
                        or (text == "DNA binding domain" and document["id"] == "7649249")
                        or document["id"] == "7649249"
                        or text == "considerable weaker suppression domain"
                        or abst["text"][int(start_ne) - title_off : int(end_ne) - title_off]
                        == "NF136,"
                    ):
                        start_ne = int(start_ne) - 1
                        end_ne = int(end_ne) - 1
                    if (
                        int(start_ne) < title_off
                        and title["text"][int(start_ne) : int(end_ne)][-1] == " "
                    ):
                        start_ne = int(start_ne) - 1
                        end_ne = int(end_ne) - 1
                    elif (
                        int(start_ne) >= title_off
                        and abst["text"][int(start_ne) - title_off : int(end_ne) - title_off][-1]
                        == " "
                    ):
                        start_ne = int(start_ne) - 1
                        end_ne = int(end_ne) - 1
                    if (
                        int(start_ne) < title_off
                        and title["text"][int(start_ne) : int(end_ne)][0] == " "
                    ):
                        start_ne = int(start_ne) + 1
                        end_ne = int(end_ne) + 1
                    elif (
                        int(start_ne) >= title_off
                        and abst["text"][int(start_ne) - title_off : int(end_ne) - title_off][0]
                        == " "
                    ):
                        start_ne = int(start_ne) + 1
                        end_ne = int(end_ne) + 1
                    if text == "Krüppel-associated box":
                        end_ne = int(end_ne) - 1
                    ann["text"] = text
                    ann["locations"] = [
                        {"offset": int(start_ne), "length": int(end_ne) - int(start_ne)}
                    ]
                    # print(ne_taple)
                    if int(start_ne) < title_off:
                        title_ne_list.append(ann)
                    else:
                        abst_ne_list.append(ann)
            elif len(line) == 0:
                title["annotations"] = title_ne_list
                abst["annotations"] = abst_ne_list
                document["passages"] = [title, abst]
                out_put_dict["documents"].append(document)
        print(ne_types.keys())
        return out_put_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    args = parser.parse_args()

    out_put_dict = pub2bc8(args.input_path + ".PubTator")
    with open(args.input_path + ".json", mode="w") as f:
        json.dump(out_put_dict, f, indent=4, ensure_ascii=False)
