# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    # 输入输出转化成序列
    def prepare_data(self, title, content):
        # 将样本中的content部分，加入cls和sep符号，将文字对应char表转成数字表示，并补齐120位
        input_seq = self.encode_sentence(content, self.config["input_max_length"], True)
        # 将样本中的title部分，加入cls，不加sep，将文字转换成数字表示，并补齐30位
        output_seq = self.encode_sentence(title, self.config["output_max_length"], True, False)  # 有起始符，无结尾符
        # 将样本中的title部分，加入sep，不加cls，将文字转换成数字表示，并补齐30位
        # gold是目标序列，input是ABCDE，output是<s>HIJK,gold是HIJK<E>，即input+部分output得到gold
        gold = self.encode_sentence(title, self.config["output_max_length"], False, True)  # 无起始符，有结尾符
        # 目标序列和输出序列错了一位，output前边有起始符<s>
        # 将三种数据封装，形成一组训练数据
        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(output_seq),
                          torch.LongTensor(gold)])

        return

    # 文本到对应的index
    # 头尾分别加入[cls]和[sep]
    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        input_id = []
        if with_cls_token:
            input_id.append(self.vocab["[CLS]"])
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if with_sep_token:
            input_id.append(self.vocab["[SEP]"])
        input_id = self.padding(input_id, max_length)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
