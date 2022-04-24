# -*- coding: utf-8 -*-
import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
from config import Config
from evaluate import Evaluator
from loader import load_data

#这个transformer是本文件夹下的代码，和我们之前用来调用bert的transformers第三方库是两回事
from transformer.Models import Transformer
#使用别人写的model文件，我们调用过来

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
1-使用transform模型，充分利用soft attention结构完成文本生成的高难度问题
2-训练阶段，基于content和output生成预测值，计算预测值和目标值gold的loss，调整transform模型
3-预测阶段，将输入文本传入模型，得到输出，是串数字，然后对照反转词表，查到对应的字符，作为输出，标题生成效果良好
"""

# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))#将配置打印出来，中文，缩进为2
    model = Transformer(config["vocab_size"], config["vocab_size"], 0, 0,
                        d_word_vec=128, d_model=128, d_inner=256,
                        n_layers=1, n_head=2, d_k=64, d_v=64,
                        )
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        # model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #加载loss
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):#32个数据是一批
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, target_seq, gold = batch_data
            pred = model(input_seq, target_seq)#分别对应loader中的input和output，然后与gold计算loss
            loss = loss_func(pred, gold.view(-1))#gold是真实值，维度是32*30
            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)


