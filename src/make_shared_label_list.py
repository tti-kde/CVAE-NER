from shared.const import task_ner_labels
import re


def separate_label_list(labels):
    separated_labels_list = []
    labels_index_list = []
    for i, label in enumerate(labels):  # separate at capital letters
        separated_label_list = re.sub(r"([A-Z]|-|_)", r" \1", label).split()
        separated_labels_list.extend(separated_label_list)
        index_list = [i] * len(separated_label_list)
        labels_index_list.extend(index_list)
    return separated_labels_list, labels_index_list


def make_shared_label_list(standard_dataset):
    separated_labels_list, labels_index_list = separate_label_list(
        task_ner_labels[standard_dataset]
    )
    shared_label_dict = {}  # {dataset_name: {label_index: standard_label}}
    for corpus_name, labels_list in task_ner_labels.items():
        if corpus_name == standard_dataset:
            continue
        shared_label_dict[corpus_name] = {0: 0}
        for j, label in enumerate(labels_list):
            for i, separated_label in enumerate(separated_labels_list):
                pattern = re.compile(separated_label, re.IGNORECASE)
                if pattern.search(label) is not None:
                    shared_label_dict[corpus_name][j + 1] = labels_index_list[i] + 1
    return shared_label_dict


if __name__ == "__main__":
    shared_label_dict = make_shared_label_list("bc8")
    print(shared_label_dict)
