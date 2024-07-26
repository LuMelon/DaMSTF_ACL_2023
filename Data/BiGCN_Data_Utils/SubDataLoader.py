import re
import numpy as np
import pandas as pd
import torch
import random
import pkuseg

def load_SST_from_file(file):
    data = []
    with open(file) as fr:
        for line in fr:
            items = line.strip("\n").split("\t")
            items.reverse()
            data.append(items)
    return data


def load_data(subj_file, obj_file):  # 5000 obj - 5000 subj | 8500 train - 500 dev - 1000 test
    def TextFile2Dataset(filepath, label):
        sents = open(filepath, encoding="latin")
        instances = [(label, line) for line in sents]
        return instances

    subj_set = TextFile2Dataset(subj_file, 0)
    obj_set = TextFile2Dataset(obj_file, 1)
    total_set = []
    total_set.extend(subj_set)
    total_set.extend(obj_set)
    total_set = random.sample(total_set, len(total_set))
    train_set = total_set[:8500]
    test_set = total_set[8500:9500]
    dev_set = total_set[9500:]
    return train_set, dev_set, test_set


def LabelSmooth(label, epsilon):
    cls_num = len(label[0])
    label = label + (epsilon) / (1.0 * (cls_num - 1))
    for i in range(len(label)):
        for j in range(len(label[0])):
            if label[i][j] > 1:
                label[i][j] -= 2 * epsilon
    return label


