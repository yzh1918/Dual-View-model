import os
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from GAT_dataloder import InMemoryWeiboDataset
from GAT_model import GAT

# ===============================================================
# 配置区
# ===============================================================

ROOT = r"D:\project\data\CED_Dataset"
TRAIN_LABEL_FILE = os.path.join(ROOT, "train_label.csv")
VAL_LABEL_FILE = os.path.join(ROOT, "val_label.csv")  # ★ 新增验证集路径
TEST_LABEL_FILE = os.path.join(ROOT, "test_label.csv")

BATCH_SIZE = 64
HIDDEN = 128
HEADS = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

MODEL_SAVE_PATH = os.path.join(ROOT, "gat_final.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# 工具函数
# ===============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1).cpu().numpy()
            y = data.y.view(-1).cpu().numpy()

            # 过滤掉 -100
            mask = (y != -100)
            if mask.sum() == 0:
                continue

            ys.append(y[mask])
            preds.append(pred[mask])

    if len(ys) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "true_p": 0.0, "true_r": 0.0, "true_f1": 0.0, "false_p": 0.0,
                "false_r": 0.0, "false_f1": 0.0}

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)

    acc = accuracy_score(ys, preds)
    macro_f1 = f1_score(ys, preds, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(ys, preds, labels=[0, 1], zero_division=0)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "true_p": p_class[0], "true_r": r_class[0], "true_f1": f1_class[0],
        "false_p": p_class[1], "false_r": r_class[1], "false_f1": f1_class[1]
    }


# ===============================================================
# 主程序
# ===============================================================

def main():
    set_seed(SEED)

    print("\n================== Loading Dataset ==================")
    dataset = InMemoryWeiboDataset(root=ROOT)
    print("Total graphs:", len(dataset))

    # 构建 filename -> dataset index 的映射
    id_to_index = {}
    for idx, fname in enumerate(dataset.processed_file_names):
        file_id = fname.replace(".pt", "")
        id_to_index[file_id] = idx

    # 读取 CSV
    train_df = pd.read_csv(TRAIN_LABEL_FILE, dtype={'id': str})
    val_df = pd.read_csv(VAL_LABEL_FILE, dtype={'id': str})  # ★ 读取验证集
    test_df = pd.read_csv(TEST_LABEL_FILE, dtype={'id': str})

    def get_data_list(df):
        data_list = []
        for graph_id, lab in zip(df["id"], df["label"]):
            if lab == -100:  # 永远舍弃无标签数据
                continue
            graph_id_str = str(graph_id)
            if graph_id_str not in id_to_index:
                continue
            data = dataset[id_to_index[graph_id]]
            data.y = torch.tensor([lab], dtype=torch.long)
            data_list.append(data)
        return data_list

    train_data_list = get_data_list(train_df)
    val_data_list = get_data_list(val_df)  # ★ 验证集列表
    test_data_list = get_data_list(test_df)

    print(f"Loaded train samples: {len(train_data_list)}")
    print(f"Loaded val samples  : {len(val_data_list)}")
    print(f"Loaded test samples : {len(test_data_list)}")

    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False)  # ★ 验证集 DataLoader
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)

    # ===============================================================
    # 构建模型
    # ===============================================================
    model = GAT(
        in_channels=dataset.num_node_features,
        hidden_channels=HIDDEN,
        out_channels=2,
        heads=HEADS,
        dropout=0.3
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print("\n================== Start Training ==================")
    # 改动1：变量名从 best_val_acc 改为 best_val_macro_f1，语义更匹配
    best_val_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            y = data.y.view(-1)

            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_examples += data.num_graphs

        avg_loss = total_loss / total_examples

        # ★ 核心改动：仅在验证集（val_loader）上进行评估和模型选拔
        val_res = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d} | TrainLoss: {avg_loss:.4f} | ValAcc: {val_res['acc']:.4f} | ValF1: {val_res['macro_f1']:.4f}")

        # 改动2：判断条件从 acc 改为 macro_f1
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]  # 改动3：更新最佳宏F1值
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # 改动4：打印提示信息同步改为 Macro-F1
            print(f">>> ⚡ Best model saved based on Validation Set (Macro-F1={best_val_macro_f1:.4f})\n")

    # ======================================================
    # ★ 终极合规步骤：加载最佳权重，跑测试集
    # ======================================================
    print("\n================== Final Test Evaluation ==================")
    # 1. 加载刚才在验证集上表现最好的那个模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # 2. 【唯一一次】运行独立测试集
    final_test_res = evaluate(model, test_loader, DEVICE)

    # 改动5：打印最佳epoch时的说明从 Acc 改为 Macro-F1
    print(f"🎉 实验完成！最佳模型选自 Epoch {best_epoch} (验证集 Macro-F1: {best_val_macro_f1:.4f})")
    print(f"该模型在【独立测试集 (Test Set)】上的最终成绩（请将以下数据填入论文表格）：")
    print(f"【全局指标】")
    print(f"   ► Accuracy  (测试集准确率) : {final_test_res.get('acc', 0):.4f}")
    print(f"   ► Macro-F1  (测试集宏F1)   : {final_test_res.get('macro_f1', 0):.4f}")
    print(f"【真言论 / True Claims (Label 0)】")
    print(f"   ► Precision (精确率)       : {final_test_res.get('true_p', 0):.4f}")
    print(f"   ► Recall    (召回率)       : {final_test_res.get('true_r', 0):.4f}")
    print(f"   ► F1-Score  (F1值)         : {final_test_res.get('true_f1', 0):.4f}")
    print(f"【假言论 / False Claims (Label 1)】")
    print(f"   ► Precision (精确率)       : {final_test_res.get('false_p', 0):.4f}")
    print(f"   ► Recall    (召回率)       : {final_test_res.get('false_r', 0):.4f}")
    print(f"   ► F1-Score  (F1值)         : {final_test_res.get('false_f1', 0):.4f}")


if __name__ == "__main__":
    main()