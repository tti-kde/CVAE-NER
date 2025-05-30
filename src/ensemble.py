import argparse
import json
import os

from sympy import li


def ensemble(zip_js_lists: list, output_js: list):
    for j, documents in enumerate(zip_js_lists):
        for i in range(1):
            annotations = []
            ano_infos = []
            ano_counts = []
            for doc in documents:
                for ano in doc["passages"][i]["annotations"]:
                    ne_type = ano["infons"]["type"]
                    offset = ano["locations"][0]["offset"]
                    length = ano["locations"][0]["length"]
                    ano_info = (offset, length, ne_type)
                    if ano_info not in ano_infos:
                        ano_infos.append(ano_info)
                        ano_counts.append([1, ano])
                    else:
                        ano_counts[ano_infos.index(ano_info)][0] += 1
            for k in range(len(ano_infos)):
                if ano_counts[k][0] > len(documents) // 2:
                    ano_offset = ano_infos[k][0]
                    ano = ano_counts[k][1]
                    annotations.append((ano_offset, ano))
            list_length = len(annotations)
            for s in range(list_length):
                for t in range(0, list_length - s - 1):
                    if annotations[t][0] > annotations[t + 1][0]:
                        temp = annotations[t]
                        annotations[t] = annotations[t + 1]
                        annotations[t + 1] = temp
            output_anos = [ano[1] for ano in annotations]
            output_js[j]["passages"][i]["annotations"] = output_anos
    return output_js


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_nums", type=str, required=True, nargs="*")
    parser.add_argument("--corpus_nums", type=str, required=True, nargs="*")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    js_lists = []
    for result_num, corpus_num in zip(args.results_nums, args.corpus_nums):
        documents = json.load(
            open(f"/workspace/data/ensemble/ent_pred_bc8_test_{corpus_num}_{result_num}.json")
        )
        js_lists.append(documents["documents"])
        if len(js_lists) == 1:
            output_js = documents

    zip_js_lists = zip(*js_lists)
    output_js = ensemble(zip_js_lists, js_lists[0])
    if not os.path.exists(args.output_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(args.output_dir)
    json.dump(
        output_js,
        open(os.path.join(args.output_dir, "ent_pred_test_ensemble.json"), "w+"),
        indent=2,
    )
