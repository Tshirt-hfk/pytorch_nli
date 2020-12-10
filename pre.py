from fastNLP import DataSet
from fastNLP import Vocabulary
from tqdm import tqdm
import spacy
import pickle
import os

train_file = ".data/snli/snli_1.0/snli_1.0_train.txt"
dev_file = ".data/snli/snli_1.0/snli_1.0_dev.txt"
test_file = ".data/snli/snli_1.0/snli_1.0_test.txt"

train_file_target = ".data/snli/train.txt"
dev_file_target = ".data/snli/dev.txt"
test_file_target = ".data/snli/test.txt"


def preData(path, target_path, lower=True):
    tokenizer = spacy.load("en_core_web_sm")
    sentence1 = []
    sentence2 = []
    gold_label = []
    with open(path, "r") as f:
        lines = f.readlines()
        headers = lines[0][:-1].split("\t")
        for line in lines[1:]:
            for header, content in zip(headers, line[:-1].split("\t")):
                if header == "sentence1":
                    sent = content.lower() if lower else content
                    sentence1.append(sent)
                elif header == "sentence2":
                    sent = content.lower() if lower else content
                    sentence2.append(sent)
                elif header in "gold_label":
                    gold_label.append(content.lower() if lower else content)

    data = {
        "sentence1": [],
        "sentence2": [],
        "target": []
    }

    for sent in tqdm(tokenizer.pipe(sentence1, n_threads=64)):
        data["sentence1"].append(" ".join([token.orth_ for token in sent]))
    for sent in tqdm(tokenizer.pipe(sentence2, n_threads=64)):
        data["sentence2"].append(" ".join([token.orth_ for token in sent]))
    data["target"] = gold_label

    with open(target_path, "w") as f:
        f.writelines("\t".join(data.keys()) + "\n")
        for s1, s2, t in zip(data["sentence1"], data["sentence2"], data["target"]):
            f.writelines(s1 + "\t" + s2 + "\t" + t + "\n")

    return data


def readData(path):
    dataset = {}
    with open(path, "r") as f:
        lines = f.readlines()
        headers = lines[0][:-1].split("\t")
        for header in headers:
            dataset[header] = []
        for line in lines[1:]:
            for header, content in zip(headers, line[:-1].split("\t")):
                dataset[header].append(content)
    return DataSet(dataset)


def loadDataset():
    train = readData(train_file)
    dev = readData(dev_file)
    test = readData(test_file)
    # 将词建立索引
    vocab = Vocabulary()
    vocab.from_dataset(train, field_name=["sentence1", "sentence2"], no_create_entry_dataset=[dev, test])
    vocab.index_dataset(train, field_name=["sentence1", "sentence2"])
    vocab.index_dataset(dev, field_name=["sentence1", "sentence2"])
    vocab.index_dataset(test, field_name=["sentence1", "sentence2"])

    # 将target转为数字
    target_vocab = Vocabulary(padding=None, unknown=None)
    target_vocab.from_dataset(train, field_name=["target"])
    target_vocab.index_dataset(train, field_name=["target"])
    target_vocab.index_dataset(dev, field_name=["target"])
    target_vocab.index_dataset(test, field_name=["target"])

    return train, dev, test, vocab, target_vocab


def preSNLI():
    preData(train_file, train_file_target)
    preData(dev_file, dev_file_target)
    preData(test_file, test_file_target)


if __name__ == "__main__":
    preSNLI()

