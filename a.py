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

vocab_file = ".data/snli/vocab.txt"
target_file = ".data/snli/target.txt"


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


def savePt(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return


def readPt(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def saveVocab(vocab, path):
    with open(path, "w") as f:
        for word, id in vocab:
            f.writelines(word + "\t" + str(id) + "\n")


def loadDataset():
    train = readData(train_file)
    dev = readData(dev_file)
    test = readData(test_file)
    # 将词建立索引
    vocab = Vocabulary()
    vocab.from_dataset(train, field_name=titles[0], no_create_entry_dataset=[dev, test])
    vocab.index_dataset(train, field_name=titles[0])
    vocab.index_dataset(dev, field_name=titles[0])
    vocab.index_dataset(test, field_name=titles[0])

    # 将target转为数字
    target_vocab = Vocabulary(padding=None, unknown=None)
    target_vocab.from_dataset(train, field_name=titles[1])
    target_vocab.index_dataset(train, field_name=titles[1])
    target_vocab.index_dataset(dev, field_name=titles[1])
    target_vocab.index_dataset(test, field_name=titles[1])

    savePt(train, train_file_pt)
    savePt(dev, dev_file_pt)
    savePt(test, test_file_pt)
    saveVocab(vocab, vocab_file)
    saveVocab(target_vocab, target_file)

    return train, dev, test


def preSNLI():
    preData(train_file, train_file_target)
    preData(dev_file, dev_file_target)
    preData(test_file, test_file_target)


if __name__ == "__main__":
    preSNLI()

