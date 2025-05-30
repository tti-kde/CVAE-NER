import json
import re

# from tabnanny import check
import numpy as np


def json2input(js: list) -> list:
    """
    入力：パス，出力：辞書入りリスト
    """
    output_list = []
    output_dict = {}
    sentences = []
    sentence_starts = []

    for document in js:
        length_doc = []
        word_len = []
        sent_len_words = []
        output_dict = {}
        output_dict["doc_key"] = document["id"]
        output_dict["sentences"] = []
        sentences = []
        ne_doc = []
        word_id2offset_doc = []
        sentences_str = document["passages"][0]["text"] + " " + document["passages"][1]["text"]
        (
            length_doc,
            sentences,
            sent_len_words,
            word_len,
            word_id2offset_doc,
        ) = separate_sentences(
            sentences_str,
            length_doc,
            sentences,
            sent_len_words,
            word_len,
            word_id2offset_doc,
        )
        output_dict["sentences"] = sentences
        output_dict["word_id2offset"] = word_id2offset_doc
        sentence_starts = len2start(length_doc)
        word_starts = len2start(word_len)
        ne_doc = [[] for _ in range(len(sentence_starts))]

        for i in range(len(document["passages"])):
            for ano in document["passages"][i]["annotations"]:
                ent = separate_on_signs(ano["text"]).split()
                idx_sent = search_position(ano["locations"][0]["offset"], sentence_starts)
                idx_word_start = search_position(ano["locations"][0]["offset"], word_starts)
                idx_word_end = len(ent) + idx_word_start - 1
                ne_tuple = (idx_word_start, idx_word_end, ano["infons"]["type"])
                ne_doc[idx_sent].append(ne_tuple)
        output_dict["ner"] = ne_doc
        output_list.append(output_dict)
    # print(count)
    return output_list


def output2bc8(batch, input_js, pred_ner, current_doc, ner_id2label_valid, other_idx=0):
    pred_ner = pred_ner[pred_ner != -100]
    num_pred = 0
    if batch == "end":
        input_js[0]["documents"][current_doc["doc_num"]]["passages"][0]["annotations"] = (
            current_doc["title_pred"]
        )
        input_js[0]["documents"][current_doc["doc_num"]]["passages"][1]["annotations"] = (
            current_doc["abst_pred"]
        )
        return input_js, current_doc

    for doc_key, tokens_len, spans, sent_start, word_id2offset in zip(
        batch["doc_key"],
        batch["tokens_len"].tolist(),
        batch["spans_word"].tolist(),
        batch["sent_start"].tolist(),
        batch["word_id2offset"].tolist(),
    ):
        if current_doc["doc_key"] != doc_key:
            if current_doc["doc_num"] != -1:
                input_js[0]["documents"][current_doc["doc_num"]]["passages"][0]["annotations"] = (
                    current_doc["title_pred"]
                )
                input_js[0]["documents"][current_doc["doc_num"]]["passages"][1]["annotations"] = (
                    current_doc["abst_pred"]
                )
                current_doc["title_pred"] = []
                current_doc["abst_pred"] = []
            current_doc["title_pred"] = []
            current_doc["abst_pred"] = []
            current_doc["doc_num"] += 1
            current_doc["doc_key"] = doc_key
            doc = input_js[0]["documents"][current_doc["doc_num"]]
            current_doc["sentences"] = doc["passages"][0]["text"] + " " + doc["passages"][1]["text"]
            current_doc["sent_for_check"] = (
                doc["passages"][0]["text"] + " " + doc["passages"][1]["text"]
            )
            current_doc["title_off"] = doc["passages"][1]["offset"]  # abstの開始位置
            current_doc["ent_num"] = 0

        for span in spans:
            if span == [0, 0, 0]:
                break
            pred = pred_ner[num_pred]
            num_pred += 1
            if pred == other_idx:
                continue
            offset = word_id2offset[span[0] - sent_start]
            length = (
                word_id2offset[span[1] - sent_start]
                - word_id2offset[span[0] - sent_start]
                + tokens_len[span[1]]
            )
            if length == 0:
                print("length is 0")
            ent = current_doc["sentences"][offset : offset + length]
            # assert ent != ".", f"{ent} is period"
            ent, offset, length = check_entity(ent, offset, length)

            length = len(ent)
            if length == 0:
                print("length is 0")
            label = ner_id2label_valid[pred]
            label = label[:-2] if label[-2] == ":" else label
            infons = {"type": label}
            locations = [{"offset": offset, "length": length}]
            predicted_ner = {
                "id": current_doc["ent_num"],
                "infons": infons,
                "text": ent,
                "locations": locations,
            }
            if offset < current_doc["title_off"]:
                current_doc["title_pred"].append(predicted_ner)
            else:
                current_doc["abst_pred"].append(predicted_ner)
            current_doc["ent_num"] += 1
    return input_js, current_doc


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
    # print(idx_max)
    position_idx = idx_max.count(idx) - 1
    return position_idx


def del_bracket(front, back, ent, offset, length):
    if re.fullmatch(rf".*\{front}", ent) is not None:
        ent = ent[0 : len(ent) - 1]
        length -= 1
    if re.fullmatch(rf"\{back}.*", ent) is not None:
        ent = ent[1 : len(ent)]
        offset += 1
    front_num = ent.count(front)
    back_num = ent.count(back)
    if re.fullmatch(rf"\{front}.*\{back}", ent) is not None:
        ent = ent[1 : len(ent) - 1]
        offset += 1
        length -= 1
    elif re.fullmatch(rf".*(\{back}|\{front})", ent) is not None and not front_num == back_num:
        ent = ent[0 : len(ent) - 1]
        length -= 1
    elif re.fullmatch(rf"(\{back}|\{front}).*", ent) is not None and not front_num == back_num:
        ent = ent[1:]
        offset += 1
    return ent, offset, length


def check_front_str(front_str, ent, offset, length):
    length_front_str = len(front_str)
    front_str = re.escape(front_str)
    if re.fullmatch(rf"{front_str}.*", ent) is not None:
        ent = ent[length_front_str:]
        offset += length_front_str
    return ent, offset, length


def check_back_str(back_str, ent, offset, length):
    length_back_str = len(back_str)
    back_str = re.escape(back_str)
    if re.fullmatch(rf".*{back_str}", ent) is not None:
        ent = ent[0 : len(ent) - length_back_str]
        length -= length_back_str
    return ent, offset, length


def check_entity(ent, offset, length):
    if re.fullmatch(r".*(\s)", ent) is not None:
        ent = ent[0 : len(ent) - 1]
        length -= 1
    if re.fullmatch(r".*-,", ent) is not None:
        ent = ent[0 : len(ent) - 2]
        length -= 2
    # ()を除去
    while True:
        ent, offset, length = del_bracket("(", ")", ent, offset, length)
        ent, offset, length = del_bracket("[", "]", ent, offset, length)
        ent, offset, length = del_bracket('"', '"', ent, offset, length)
        ent, offset, length = del_bracket("'", "'", ent, offset, length)
        ent, offset, length = del_bracket("(", ")", ent, offset, length)
        if (
            re.fullmatch(r".*(-|\+)", ent) is not None
            and re.search(r"(I|\s|Ca2|O2)(-|\+)", ent) is None
        ):
            ent = ent[0 : len(ent) - 1]
            length -= 1
        elif re.fullmatch(r".*\.,", ent) is not None:
            ent = ent[0 : len(ent) - 1]
            length -= 1
            break
        elif re.fullmatch(r".*(:|;|,|\.|/)", ent) is not None:
            ent = ent[0 : len(ent) - 1]
            length -= 1
        else:
            break
    ent, offset, length = check_back_str("?", ent, offset, length)
    ent, offset, length = check_back_str("->", ent, offset, length)
    ent, offset, length = check_back_str(">", ent, offset, length)
    ent, offset, length = check_back_str("<", ent, offset, length)
    ent, offset, length = check_back_str("*5", ent, offset, length)
    ent, offset, length = check_back_str("*15", ent, offset, length)
    ent, offset, length = check_back_str("*1b", ent, offset, length)
    ent, offset, length = check_back_str("(+25", ent, offset, length)
    if re.fullmatch(r"CYP2F1\*.*", ent) is not None:
        length -= len(ent)
        ent = "CYP2F1"
        length -= len(ent)
    ent, offset, length = check_front_str("AUC", ent, offset, length)
    ent, offset, length = check_front_str("/", ent, offset, length)
    ent, offset, length = check_front_str("rt", ent, offset, length)
    ent, offset, length = check_front_str("*15+", ent, offset, length)
    ent, offset, length = check_front_str("3+", ent, offset, length)
    return ent, offset, length


def separate_on_signs(sent):
    for _ in range(2):
        sent = re.sub(
            "([0-9a-zA-Z]|\)|\.|/|\+|-)(-|\(|\)|/|;|:|\.|%|'|>|<|\[|\])(\(|-|\+|\.|/|\*|[0-9a-zA-Z])",
            "\\1 \\3",
            sent,
        )
        sent = re.sub("([a-zA-Z])\+([a-zA-Z])", "\\1 \\2", sent)
        sent = re.sub("([0-9a-zA-Z]|\))--([0-9a-zA-Z])", "\\1  \\2", sent)
        sent = re.sub("([0-9a-zA-Z]),([0-9a-zA-Z])", "\\1 \\2", sent)
        sent = re.sub("([0-9a-zA-Z])\((\+|-)\)(\)|[0-9a-zA-Z])", "\\1 + \\3", sent)
        sent = re.sub("([0-9a-zA-Z])\((\+|-)/(\+|-|lo)\)(\s|.)", "\\1 \\2/\\3 \\4", sent)
        sent = re.sub("(\))\.([0-9a-zA-Z])", "\\1 \\2", sent)
    return sent


def separate_sentences(
    doc,
    length_sent_sum,
    sent_list,
    length_words_in_sent,
    length_words_in_doc,
    word_id2offset_doc=None,
):
    doc = re.sub("(et al|e\.g|i\.e|cf)\.", "\\1:", doc)
    doc = re.sub("p. Glu104X", "p- Glu104X", doc)
    doc = re.sub("p. Pro", "p- Pro", doc)
    doc = re.sub("p. Gln", "p- Gln", doc)
    doc = re.sub("p. Asp", "p- Asp", doc)
    doc_sent = re.sub("\.\s([A-Z])", "\t\\1", doc)
    # -, /を除去
    doc_sent = separate_on_signs(doc_sent)
    doc_sent = separate_on_signs(doc_sent)
    doc_list = [
        sentence for sentence in re.split("(\t)|(\.$)|(\n)", doc_sent) if sentence is not None
    ]

    text_idx = 0
    for i, sentence in enumerate(doc_list):
        if re.fullmatch(r"(\.)|(\t)|(\n)|(\s*\x0c*\s*)", sentence) is not None:
            continue
        # print(sentence)
        length_sent_sum.append(len(sentence) + 2)
        words = sentence.split()
        assert len(words) != 0, i
        word_id2offset = []
        sent_start = text_idx
        sentence += "."
        if word_id2offset_doc is not None:
            if words[-1] == ".":
                words = words[:-1]
            for idx, word in enumerate(words + ["."]):
                while True:
                    if sentence[text_idx - sent_start : text_idx - sent_start + len(word)] == word:
                        word_id2offset.append(text_idx)
                        text_idx += len(word)
                        break
                    text_idx += 1
                if word == ".":
                    text_idx += 1
            word_id2offset_doc.append(word_id2offset)
        sentence_list = words + ["."]
        sent_list.append(sentence_list)
        offsets = word_id2offset  # list(word_id2offset.values())
        word_len_sent = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        word_len_sent.append(2)
        length_words_in_doc += word_len_sent
        length_words_in_sent.append(len(sentence_list))
    return (
        length_sent_sum,
        sent_list,
        length_words_in_sent,
        length_words_in_doc,
        word_id2offset_doc,
    )
