import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path


class Config_base(object):
    """配置参数"""

    def __init__(self, model_name, task_name):

        # path
        self.model_name = model_name
        self.task_name = task_name

        # --- MODIFIED: 使用相对路径以保证跨平台兼容性 ---
        project_root = path.dirname(path.dirname(path.abspath(__file__)))
        self.clip_path = path.join(project_root, 'model', 'chinese-clip-vit-base-patch16')
        self.roberta_path = path.join(project_root, 'model', 'chinese-roberta-wwm-ext')
        self.bert_path = path.join(project_root, 'model', 'bert-base-chinese')
        self.vit_path = path.join(project_root, 'model', 'vit-base-patch16-224')
        self.resnet_path = path.join(project_root, 'model', 'resnet-50')

        if self.model_name == "clip":
            self.hidden_dim = 512
        else:
            self.hidden_dim = 768

        self.meme_path = path.join(project_root, 'data', 'meme')
        self.train_path = path.join(project_root, 'data', 'train_data_discription.json')
        self.dev_path = path.join(project_root, 'data', 'test_data_discription.json')
        self.test_path = path.join(project_root, 'data', 'test_data_discription.json')
        self.result_path = path.join(project_root, 'result')
        self.checkpoint_path = path.join(project_root, 'saved_dict')
        self.data_path = self.checkpoint_path + '/' + self.model_name + '_data.tar'

        if self.task_name == "task_1":
            self.num_classes = 2  # 类别数
        else:
            self.num_classes = 5  # 类别数

        # dataset
        self.seed = 1
        self.pad_size = 64  # 每句话处理成的长度(短填长切)

        # model
        self.dropout = 0.5  # 随机失活
        self.fc_hidden_dim = 256
        self.weight = 0.5

        # --- NEW: 为新的跨模态融合模块添加超参数 ---
        self.fusion_num_heads = 12  # 交叉注意力头的数量
        self.fusion_dropout = 0.1  # 融合模块内部的dropout率

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
        self.learning_rate = 1e-5  # 学习率
        self.num_epochs = 10  # epoch数
        self.num_warm = 0  # 预热
        self.batch_size = 32  # mini-batch大小

        # evaluate
        self.score_key = "F1"  # 评价指标