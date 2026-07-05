# BERT_train.py —— IDE 直接运行版（无 argparse）

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import nn
from BERT_loaddata import model  # 你的 BERT 模型构造函数
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ======================================================
# 配置（你只需修改这几项）
# ======================================================
JSON_DIR = r"D:\project\data\PHEME_Dataset\original-microblog"
LABEL_PATH = r"D:\project\data\PHEME_Dataset\train_label.csv"  # 用于定位目录即可

TOKENIZER_PATH = r"D:/bert-base-uncased"
MAX_LENGTH = 128

BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 10
SEED = 1234

MODEL_SAVE_PATH = r"D:\project\data\PHEME_Dataset\bert_final.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ======================================================
# 固定随机种子
# ======================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ======================================================
# Dataset
# ======================================================
class TextDataset(Dataset):
    def __init__(self, ids, labels, tokenizer, json_dir, max_length):
        self.ids = ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.json_dir = json_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        label = self.labels[i]

        with open(os.path.join(self.json_dir, f"{sid}.json"), "r", encoding="utf8") as f:
            j = json.load(f)

        text = j["text"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]).squeeze(0)),
            torch.tensor(label, dtype=torch.long)
        )


# ======================================================
# 形状修复与评估辅助函数
# ======================================================
def fix_shapes(input_ids, attn, ttids):
    input_ids = input_ids.long()
    attn = attn.long()
    ttids = ttids.long()

    if ttids.dim() == 3:
        if ttids.size(1) == 1:
            ttids = ttids.squeeze(1)
        elif ttids.size(2) == 1:
            ttids = ttids.squeeze(2)
        else:
            ttids = ttids[..., 0]

    if ttids.dim() != 2:
        raise RuntimeError(f"token_type_ids shape 异常: {ttids.shape}")

    return input_ids, attn, ttids


def evaluate(net, loader, device):
    """通用评估函数，用于验证集和测试集"""
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attn, ttids, labels in loader:
            input_ids, attn, ttids = fix_shapes(input_ids, attn, ttids)

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            ttids = ttids.to(device)

            outputs = net(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    if len(all_labels) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "true_p": 0.0, "true_r": 0.0, "true_f1": 0.0, "false_p": 0.0,
                "false_r": 0.0, "false_f1": 0.0}

    # 1. 计算全局指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 2. 计算每个类别的具体指标
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1],
                                                                    zero_division=0)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "true_p": p_class[0], "true_r": r_class[0], "true_f1": f1_class[0],
        "false_p": p_class[1], "false_r": r_class[1], "false_f1": f1_class[1]
    }


# ======================================================
# 数据加载提取函数 (复用逻辑)
# ======================================================
def load_data_from_csv(csv_path, json_dir):
    df = pd.read_csv(csv_path, dtype={'id': str})
    ids = []
    labels = []
    miss_count = 0

    for graph_id, lab in zip(df["id"], df["label"]):
        # ★ 严格舍弃 -100
        if lab == -100:
            continue
        uid = str(graph_id)
        try:
            label_int = int(lab)
        except ValueError:
            continue

        json_path = os.path.join(json_dir, f"{uid}.json")
        if os.path.exists(json_path):
            ids.append(uid)
            labels.append(label_int)
        else:
            miss_count += 1

    return ids, np.array(labels, dtype=int), miss_count, len(df)


# ======================================================
# 加载 训练集、验证集、测试集
# ======================================================
train_label_path = os.path.join(os.path.dirname(LABEL_PATH), "train_label.csv")
val_label_path = os.path.join(os.path.dirname(LABEL_PATH), "val_label.csv")  # ★ 新增验证集路径
test_label_path = os.path.join(os.path.dirname(LABEL_PATH), "test_label.csv")

print("\n================== Loading Datasets ==================")
train_ids, train_labels, miss_train, total_train = load_data_from_csv(train_label_path, JSON_DIR)
print(f"训练集 CSV行数: {total_train} | 有效样本数: {len(train_ids)} | 缺失 JSON 数: {miss_train}")

val_ids, val_labels, miss_val, total_val = load_data_from_csv(val_label_path, JSON_DIR)
print(f"验证集 CSV行数: {total_val} | 有效样本数: {len(val_ids)} | 缺失 JSON 数: {miss_val}")

test_ids, test_labels, miss_test, total_test = load_data_from_csv(test_label_path, JSON_DIR)
print(f"测试集 CSV行数: {total_test} | 有效样本数: {len(test_ids)} | 缺失 JSON 数: {miss_test}")

# ======================================================
# DataLoader
# ======================================================
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

train_dataset = TextDataset(train_ids, train_labels, tokenizer, JSON_DIR, MAX_LENGTH)
val_dataset = TextDataset(val_ids, val_labels, tokenizer, JSON_DIR, MAX_LENGTH)
test_dataset = TextDataset(test_ids, test_labels, tokenizer, JSON_DIR, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # ★ 验证集 DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# 模型 / 优化器
# ======================================================
net = model(r"D:/bert-base-uncased", num_classes=2).to(DEVICE)
optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ======================================================
# 训练开始前打印一次形状
# ======================================================
for batch in train_loader:
    input_ids, attn, ttids, labels = batch
    print("\ninput_ids shape:", input_ids.shape)
    print("attn shape:", attn.shape)
    print("ttids shape:", ttids.shape)
    break

# ======================================================
# 训练
# ======================================================
print("\n================== Start Training ==================")
best_val_macro_f1 = -1.0  # ★ 改为追踪最佳 Macro-F1
best_epoch = -1

for epoch in range(1, EPOCHS + 1):

    net.train()
    running_loss = 0
    running_corr = 0
    running_total = 0

    for b, (input_ids, attn, ttids, labels) in enumerate(train_loader):
        input_ids, attn, ttids = fix_shapes(input_ids, attn, ttids)

        input_ids = input_ids.to(DEVICE)
        attn = attn.to(DEVICE)
        ttids = ttids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = net(input_ids=input_ids, attention_mask=attn, token_type_ids=ttids)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct = (preds == labels).sum().item()

        running_loss += loss.item() * labels.size(0)
        running_corr += correct
        running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_corr / running_total

    # ★ 核心改动：在验证集上评估
    val_res = evaluate(net, val_loader, DEVICE)

    print(
        f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_res['acc']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f}")

    # ★ 核心升级：严格基于验证集的 Macro-F1 作为保存最佳模型的“裁判”
    if val_res['macro_f1'] > best_val_macro_f1:
        best_val_macro_f1 = val_res['macro_f1']
        best_epoch = epoch
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
        print(f">>> Best model saved based on Validation Set (Macro-F1={best_val_macro_f1:.4f})\n")

# ======================================================
# ★ 训练大循环结束后的最终总结打印（测试集评估）
# ======================================================
print("\n================== Final Test Evaluation ==================")
# 1. 加载验证集挑选出来的最佳权重
net.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 2. 在独立测试集上进行最终评价
test_res = evaluate(net, test_loader, DEVICE)

print(f"🎉 完美收官！最佳模型在验证集上的最佳 Epoch 是 {best_epoch} (Val Macro-F1: {best_val_macro_f1:.4f})")
print(f"该模型在【独立测试集 (Test Set)】上的细粒度指标如下：")
print(f"【全局指标】")
print(f"   ► Accuracy  (测试集准确率) : {test_res.get('acc', 0):.4f}")
print(f"   ► Macro-F1  (测试集宏F1)   : {test_res.get('macro_f1', 0):.4f}")
print(f"【真言论 / True Claims (Label 0)】")
print(f"   ► Precision (精确率)       : {test_res.get('true_p', 0):.4f}")
print(f"   ► Recall    (召回率)       : {test_res.get('true_r', 0):.4f}")
print(f"   ► F1-Score  (F1值)         : {test_res.get('true_f1', 0):.4f}")
print(f"【假言论 / False Claims (Label 1)】")
print(f"   ► Precision (精确率)       : {test_res.get('false_p', 0):.4f}")
print(f"   ► Recall    (召回率)       : {test_res.get('false_r', 0):.4f}")
print(f"   ► F1-Score  (F1值)         : {test_res.get('false_f1', 0):.4f}")