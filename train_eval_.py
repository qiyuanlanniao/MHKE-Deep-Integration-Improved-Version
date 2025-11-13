import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import time
import json
from dataset.dataset import get_time_dif, convert_onehot
# --- MODIFIED: 导入新旧所有模型 ---
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *


def train(config, train_iter, dev_iter):
    # --- MODIFIED: 这里的模型实例化现在会调用我们改进后的新模型 ---
    if config.model_name == "clip":
        model = MHKE_CLIP(config).to(config.device)
    elif config.model_name == "vit-roberta":
        model = VitRobertaMemesClassifier(config).to(config.device)
    elif config.model_name == "vit":
        model = VitClassifier(config).to(config.device)
    elif config.model_name == "resnet":
        model = ResNetClassifier(config).to(config.device)
    elif config.model_name == "roberta":
        model = RobertaClassifier(config).to(config.device)
    elif config.model_name == "bert":
        model = BertClassifier(config).to(config.device)
    elif config.model_name == "MHKE":
        model = MHKE(config).to(config.device)

    # --- ADDED FOR FALLBACK ---
    else:
        # 添加一个默认或错误处理
        raise ValueError(f"Model name '{config.model_name}' is not recognized.")

    model_name = '{}_B-{}_E-{}_Lr-{}_w-{}_{}_add_Fusion'.format(config.model_name, config.batch_size,
                                                                config.num_epochs, config.learning_rate, config.weight,
                                                                config.task_name)

    params = list(model.named_parameters())

    if config.model_name == "resnet":
        model_optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate)
    else:
        model_optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate)

    # --- MODIFIED: 关键改动 - 更换损失函数 ---
    # 从BCEWithLogitsLoss改为CrossEntropyLoss，更适合多分类任务
    loss_fn = nn.CrossEntropyLoss()

    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []

        for batch in tqdm(train_iter, desc='Training', colour='MAGENTA'):
            model.zero_grad()
            logit = model(**batch)

            # --- MODIFIED: 关键改动 - 调整标签格式以适应新的损失函数 ---
            if config.task_name == "task_1":
                # one-hot 格式的标签，用于计算指标
                label_onehot = batch['label']
                # CrossEntropyLoss 需要类别索引 (LongTensor)
                label_index = torch.argmax(label_onehot, dim=1).to(config.device)
            else:  # task_2
                label_onehot = batch['type_label']
                label_index = torch.argmax(label_onehot, dim=1).to(config.device)

            # 使用新的损失函数计算 loss
            loss = loss_fn(logit, label_index)

            # --- MODIFIED: 预测逻辑现在基于 logits ---
            # get_preds 和 get_preds_task2 内部逻辑也需要微调
            if config.task_name == "task_1":
                pred = get_preds(config, logit.cpu())
            else:
                pred = get_preds_task2(config, logit.cpu())

            preds.extend(pred)
            labels.extend(label_onehot.detach().cpu().numpy())

            loss_all += loss.item()
            # model_optimizer.zero_grad() # 这行是多余的，前面已经有了
            loss.backward()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time) / 60.))
        print("TRAINED for {} epochs".format(epoch))

        if epoch >= config.num_warm:
            trn_scores = get_scores(preds, labels, loss_all, len(train_iter), data_name="TRAIN")
            dev_scores, _ = eval(config, model, loss_fn, dev_iter, data_name='DEV')
            with open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a') as f:
                f.write(
                    ' ==================================================  Epoch: {}  ==================================================\n'.format(
                        epoch))
                f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores)))
            max_score = save_best(config, epoch, model_name, model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))


def eval(config, model, loss_fn, dev_iter, data_name='DEV'):
    model.eval()  # --- ADDED: 确保模型在评估模式 ---
    loss_all = 0.
    preds = []
    labels = []

    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            logit = model(**batch)

            # --- MODIFIED: 同样调整标签格式 ---
            if config.task_name == "task_1":
                label_onehot = batch['label']
                label_index = torch.argmax(label_onehot, dim=1).to(config.device)
                pred = get_preds(config, logit.cpu())
            else:  # task_2
                label_onehot = batch['type_label']
                label_index = torch.argmax(label_onehot, dim=1).to(config.device)
                pred = get_preds_task2(config, logit.cpu())

            loss = loss_fn(logit, label_index)

            preds.extend(pred)
            labels.extend(label_onehot.detach().cpu().numpy())
            loss_all += loss.item()

    dev_scores = get_scores(preds, labels, loss_all, len(dev_iter), data_name=data_name)
    return dev_scores, preds


def test(model, dev_iter):
    # ... (test 函数保持不变，因为它不计算损失) ...
    model.eval()
    preds = []
    labels = []

    for batch in tqdm(dev_iter, desc='Testing', colour='CYAN'):
        with torch.no_grad():
            logit = model(**batch).cpu()
            label = batch['label']
            pred = output_preds(logit)

            preds.extend(pred)
            labels.extend(label.detach().numpy())

        df = pd.DataFrame({'new_pred': preds})
        output_file = 'preds.csv'
        df.to_csv(output_file, index=False)
    return preds


# Task 1: Harmful Meme Detection (无需修改)
def get_preds(config, logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        result = convert_onehot(config, result)
        new_results.append(result)
    return new_results


# Task 2: Harmful Type Discrimination
def get_preds_task2(config, logit):
    all_results = []
    # --- MODIFIED: 移除了 sigmoid, 直接在logits上操作 ---
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    for i in range(len(results)):
        result = convert_onehot(config, results[i])
        all_results.append(result)
    return all_results


# ... (文件的其余部分保持不变) ...
def output_preds(logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        new_results.append(result)
    return new_results


def get_scores(all_preds, all_lebels, loss_all, length, data_name):
    score_dict = dict()
    f1 = f1_score(all_lebels, all_preds, average='macro')
    acc = accuracy_score(all_lebels, all_preds)
    all_f1 = f1_score(all_lebels, all_preds, average=None)
    pre = precision_score(all_lebels, all_preds, average='macro')
    recall = recall_score(all_lebels, all_preds, average='macro')

    score_dict['F1'] = f1
    score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall

    score_dict['all_loss'] = loss_all / length
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items():
        print("{}: {}".format(s_name, s_val))
    return score_dict


def save_best(config, epoch, model_name, model, score, max_score):
    score_key = config.score_key
    curr_score = score[score_key]
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(
        epoch, score_key, curr_score, score_key, max_score))

    if curr_score > max_score or epoch == 0:
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        return curr_score
    else:
        return max_score
