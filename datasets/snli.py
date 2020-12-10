import os
import torch
from torchtext.data import Field, Iterator
from torchtext import datasets
from utils import makedirs
from fastNLP import DataSet, RandomSampler, DataSetIter
from fastNLP import Vocabulary

__all__ = ['snli']

train_file_target = ".data/snli/train.txt"
dev_file_target = ".data/snli/dev.txt"
test_file_target = ".data/snli/test.txt"


def readData(path):
    dataset = {}
    with open(path, "r") as f:
        lines = f.readlines()
        headers = lines[0][:-1].split("\t")
        for header in headers:
            dataset[header] = []
        for line in lines[1:]:
            data = line[:-1].split("\t")
            dataset[headers[0]].append(data[0].split())
            dataset[headers[1]].append(data[1].split())
            dataset[headers[2]].append(data[2])

    return DataSet(dataset)


class SNLI():
    def __init__(self, options):

        self.train = readData(train_file_target)
        self.dev = readData(dev_file_target)
        self.test = readData(test_file_target)
        # 将词建立索引
        self.vocab = Vocabulary()
        self.vocab.from_dataset(self.train, field_name=["premise", "hypothesis"], no_create_entry_dataset=[self.dev, self.test])
        self.vocab.index_dataset(self.train, field_name=["premise", "hypothesis"])
        self.vocab.index_dataset(self.dev, field_name=["premise", "hypothesis"])
        self.vocab.index_dataset(self.test, field_name=["premise", "hypothesis"])

        # 将target转为数字
        self.target_vocab = Vocabulary(padding=None, unknown=None)
        self.target_vocab.from_dataset(self.train, field_name=["target"])
        self.target_vocab.index_dataset(self.train, field_name=["target"])
        self.target_vocab.index_dataset(self.dev, field_name=["target"])
        self.target_vocab.index_dataset(self.test, field_name=["target"])

        self.train.set_input("premise", "hypothesis")
        self.train.set_target("target")
        self.dev.set_input("premise", "hypothesis")
        self.dev.set_target("target")
        self.test.set_input("premise", "hypothesis")
        self.test.set_target("target")

        self.train_iter = DataSetIter(dataset=self.train, batch_size=options['batch_size'], sampler=RandomSampler())
        self.dev_iter = DataSetIter(dataset=self.dev, batch_size=options['batch_size'])
        self.test_iter = DataSetIter(dataset=self.test, batch_size=options['batch_size'])

    def vocab_size(self):
        return len(self.vocab)

    def out_dim(self):
        return len(self.target_vocab)

    def labels(self):
        return self.target_vocab.word2idx


def snli(options):
    return SNLI(options)

