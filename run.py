import os
import numpy as np

import torch
import torch.nn as nn
import json

from config.Config_base import Config_base
from dataset.dataset import MemeDataset, DataLoader
from train_eval_ import train, test

# --- MODIFIED: 确保导入所有模型 ---
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *

if __name__ == '__main__':

    # --- MODIFIED: 切换到新的融合模型进行训练 ---
    # model_name = "clip"  # 这会运行改进后的 MHKE_CLIP
    model_name = "MHKE"  # 这会运行改进后的 MHKE (RoBERTa+ViT)

    # 您也可以继续运行旧的基础模型进行对比，例如：
    # model_name = "vit-roberta" 

    task_name = "task_2"
    config = Config_base(model_name, task_name)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(config.data_path):
        print(f"Creating and saving dataset to {config.data_path}...")
        trn_data = MemeDataset(config, training=True)
        test_data = MemeDataset(config, training=False)
        torch.save({
            'trn_data': trn_data,
            'test_data': test_data,
        }, config.data_path)
        print("Dataset saved.")
    else:
        print(f"Loading dataset from {config.data_path}...")
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        test_data = checkpoint['test_data']
        print("Dataset loaded.")

    print('The size of the Training dataset: {}'.format(len(trn_data)))
    print('The size of the Test dataset: {}'.format(len(test_data)))

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=True)  # 建议训练时 shuffle=True
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    train(config, train_iter, test_iter)
