# -*- coding: utf-8 -*-
import torch
import collections
import io
import json
import six
import sys
import argparse
from loader import load_data
from collections import defaultdict, OrderedDict

from transformer.Translator import Translator

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        #用同样的load方法，将sample的103样本作为检测样本
        self.valid_data = load_data(config["valid_data_path"], config, logger, shuffle=False)
        #反转词表
        self.reverse_vocab = dict([(y, x) for x, y in self.valid_data.dataset.vocab.items()])
        self.translator = Translator(self.model,
                                     config["beam_size"],
                                     config["output_max_length"],
                                     config["pad_idx"],
                                     config["pad_idx"],
                                     config["start_idx"],
                                     config["end_idx"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int)  # 用于存储测试结果
        for index, batch_data in enumerate(self.valid_data):
            input_seqs, target_seqs, gold = batch_data
            for input_seq in input_seqs:
                #接收输入的数字，通过模型生成输出数字
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
                print("输入：", self.decode_seq(input_seq))
                print("输出：", self.decode_seq(generate))
                break#32个样本一组，每组数据输出一个即可，
        return

    def decode_seq(self, seq):
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])


