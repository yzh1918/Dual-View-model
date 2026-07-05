import os
import json
import warnings
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer

# Project imports
from BERT_loaddata import model as BertModelClass
from GAT_dataloder import InMemoryWeiboDataset
from GAT_model import GAT
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch as GraphBatch

# ======================================================
# 导入自定义工具模块 (外置的函数都在这里)
# ======================================================
from utils_mix import *

# ------------------ 配置区 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"D:\project\data\CED_Dataset"
JSON_ROOT = r"D:\project\data\CED_Dataset\original-microblog"
EVAL_LABEL_CSV = os.path.join(ROOT, "test_label.csv")
TRAIN_LABEL_CSV = os.path.join(ROOT, "train_label.csv")
VAL_LABEL_CSV = os.path.join(ROOT, "val_label.csv")
LABEL_PATH = os.path.join(ROOT, "label.csv")

BERT_WEIGHT = os.path.join(ROOT, "bert_final.pt")
GAT_WEIGHT = os.path.join(ROOT, "gat_final.pt")
BERT_PREHEATED_PATH = os.path.join(ROOT, "bert_final.pt")
GAT_PREHEATED_PATH = os.path.join(ROOT, "gat_final.pt")

TOKENIZER_PATH = r"D:/bert-base-chinese"

BATCH_SIZE_BERT = 16
BATCH_SIZE_GAT = 8
MAX_LEN = 128

# Stage2 Hyperparams
BATCH_SIZE = 16
EPOCHS = 10
BERT_LR = 2e-5
GAT_LR_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-4
WEIGHT_DECAY_bert= 2e-5
# === 核心策略参数 ===
CONSENSUS_THRESH = 0.85
RESCUE_THRESH = 0.98

# === FixMatch 参数 ===
LAMBDA_FIXMATCH = 1.0  # 一致性 Loss 权重
DROP_EDGE_RATE = 0.2  # GAT 强增强剪枝率

DROP_TAIL_STAGE2 = True
DO_STAGE2 = True
STRONG_AUG_MODE = {
    "bert": "identity",   # 选项: "shuffle" | "identity"
    "gat": "identity",   # 选项: "dropedge" | "identity"
    "drop_rate": 0.2     # 仅 dropedge 模式生效，可改为 0.1 测试 G5
}
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------ 数据集定义 (保留在主文件以便 dataloader 调用) ------------------
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)


class BertAlignedDataset(Dataset):
    def __init__(self, label_df, json_root, tokenizer, max_length=128):
        self.df = label_df
        self.json_root = json_root
        self.tk = tokenizer
        self.maxlen = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["id"])
        lab = int(row["label"])
        json_path = os.path.join(self.json_root, sid + ".json")
        if not os.path.exists(json_path): return None
        try:
            with open(json_path, "r", encoding="utf8") as f:
                j = json.load(f)
        except:
            return None
        enc = self.tk(j.get("text", ""), truncation=True, padding="max_length", max_length=self.maxlen,
                      return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(lab, dtype=torch.long)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return default_collate(batch)


# ------------------ Stage 1 初始化 ------------------
eval_df = read_label_csv(EVAL_LABEL_CSV)
bert_dataset = BertAlignedDataset(eval_df, JSON_ROOT, tokenizer, max_length=MAX_LEN)
bert_loader = DataLoader(bert_dataset, batch_size=BATCH_SIZE_BERT, shuffle=False, collate_fn=safe_collate)

graph_dataset = InMemoryWeiboDataset(root=ROOT)
graph_map = {str(getattr(graph_dataset[i], "sample_id", "")): graph_dataset[i] for i in range(len(graph_dataset)) if
             str(getattr(graph_dataset[i], "sample_id", ""))}

aligned_graphs = []
for _, row in eval_df.iterrows():
    sid = str(row["id"])
    if sid in graph_map:
        d = graph_map[sid]
        d.y = torch.tensor([int(row["label"])], dtype=torch.long)
        aligned_graphs.append(d)
gat_loader = GeoDataLoader(aligned_graphs, batch_size=BATCH_SIZE_GAT, shuffle=False)

# 模型加载
bert_model = BertModelClass(r"D:/bert-base-chinese", num_classes=2).to(DEVICE)
bert_model.load_state_dict(torch.load(BERT_WEIGHT, map_location=DEVICE), strict=False)
bert_model.eval()

in_dim = graph_dataset.num_node_features
gat_model = GAT(in_channels=in_dim, hidden_channels=128, out_channels=2, heads=8, dropout=0.3).to(DEVICE)
gat_model.load_state_dict(torch.load(GAT_WEIGHT, map_location=DEVICE), strict=False)
gat_model.eval()


def run_eval_bert():
    total = 0;
    correct = 0
    if bert_loader is None: return 0.0
    with torch.no_grad():
        for batch in bert_loader:
            if batch is None: continue
            input_ids, attn_mask, labels = batch
            input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            preds = get_logits(bert_model(input_ids, attn_mask)).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"[BERT] aligned eval accuracy = {correct / total:.4f}")


def run_eval_gat():
    total = 0;
    correct = 0
    with torch.no_grad():
        for batch in gat_loader:
            batch = batch.to(DEVICE)
            preds = get_logits(gat_model(batch.x, batch.edge_index, batch.batch)).argmax(dim=1)
            labels = batch.y.view(-1).to(DEVICE)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"[GAT] aligned eval accuracy = {correct / total:.4f}")


# ------------------ Main ------------------
def main():
    print("============ Start Stage 1 Evaluation ============")
    run_eval_bert()
    run_eval_gat()
    print("============ Stage 1 Done ============")
    if not DO_STAGE2: return

    print("\n============ Preparing Stage 2 (FixMatch: Consensus + Rescue) ============")
    common_ids, bert_ids, gat_filenames = build_common_alignment(JSON_ROOT, graph_dataset)

    # 索引映射
    id_to_global = {sid: i for i, sid in enumerate(common_ids)}
    global_to_bert_idx = {id_to_global[sid]: i for i, sid in enumerate(bert_ids) if sid in id_to_global}
    global_to_gat_idx = {id_to_global[sid]: i for i, sid in enumerate(gat_filenames) if sid in id_to_global}

    # 标签加载
    df_list = []
    for p in [TRAIN_LABEL_CSV, EVAL_LABEL_CSV, VAL_LABEL_CSV]:
        if os.path.exists(p): df_list.append(pd.read_csv(p, dtype=str))
    df_label = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    label_map = {}
    if not df_label.empty:
        for _, r in df_label.iterrows():
            try:
                label_map[str(r[0]).strip()] = int(float(str(r[1]).strip()))
            except:
                pass
    global_labels = [label_map.get(sid, -100) for sid in common_ids]

    # 数据集划分
    test_ids = set(str(i) for i in eval_df["id"])
    val_ids = set(str(i) for i in read_label_csv(VAL_LABEL_CSV)["id"])

    test_indices = [id_to_global[sid] for sid in common_ids if sid in test_ids]
    val_indices = [id_to_global[sid] for sid in common_ids if sid in val_ids]
    train_indices = [i for i in range(len(common_ids)) if
                     common_ids[i] not in test_ids and common_ids[i] not in val_ids]

    if DROP_TAIL_STAGE2:
        test_indices = test_indices[:(len(test_indices) // BATCH_SIZE) * BATCH_SIZE]
        val_indices = val_indices[:(len(val_indices) // BATCH_SIZE) * BATCH_SIZE]

    # 训练准备
    bert_label_rows = [{"id": sid, "label": label_map.get(sid, -100)} for sid in common_ids]
    bert_dataset_full = BertAlignedDataset(pd.DataFrame(bert_label_rows), JSON_ROOT, tokenizer, max_length=MAX_LEN)
    gat_dataset_local = graph_dataset

    # 用于全局追踪所有 Alpha 的最佳结果
    global_best_alpha = 0.1
    global_best_val_macro_f1 = -1.0
    global_best_test_metrics = {}

    # ========================== Alpha 网格搜索外循环 ==========================
    for alpha_step in range(3, 4):
        current_alpha = alpha_step / 10.0
        print("\n" + "=" * 60)
        print(f"🚀 [Alpha 步进搜索] 开始测试 Alpha (BERT权重) = {current_alpha:.1f}")
        print("=" * 60)

        # 每次切换 Alpha，必须重新加载预训练权重，避免模型延续上一次 Alpha 的训练状态！
        bert_model_stage2 = BertModelClass(r"D:/bert-base-chinese", num_classes=2).to(DEVICE)
        gat_model_stage2 = GAT(in_channels=in_dim, hidden_channels=128, out_channels=2, heads=8).to(DEVICE)

        if os.path.exists(BERT_PREHEATED_PATH):
            bert_model_stage2.load_state_dict(torch.load(BERT_PREHEATED_PATH, map_location=DEVICE), strict=False)
        if os.path.exists(GAT_PREHEATED_PATH):
            gat_model_stage2.load_state_dict(torch.load(GAT_PREHEATED_PATH, map_location=DEVICE), strict=False)

        bert_opt = torch.optim.AdamW(bert_model_stage2.parameters(), lr=BERT_LR, weight_decay=WEIGHT_DECAY_bert)
        gat_opt_stage2 = torch.optim.AdamW(gat_model_stage2.parameters(), lr=GAT_LR_STAGE2, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1).to(DEVICE)

        unlabeled_pool = [i for i in train_indices if global_labels[i] == -100]
        labeled_pool = [i for i in train_indices if global_labels[i] != -100]

        best_val_macro_f1_this_alpha = -1.0
        best_epoch_this_alpha = 0

        # ========================== 内循环：当前 Alpha 下的完整 Epoch 训练 ==========================
        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Alpha {current_alpha:.1f} | Stage2 Epoch {epoch}/{EPOCHS} ---")

            # 1. 伪标签生成 (Consensus + Rescue) - 不再传入 alpha
            pseudo_labels = generate_pseudo_fixed_threshold(
                bert_model_stage2, gat_model_stage2, unlabeled_pool,
                global_to_bert_idx, global_to_gat_idx,
                bert_dataset_full, gat_dataset_local, DEVICE,
                batch_size=BATCH_SIZE,
                consensus_thresh=CONSENSUS_THRESH,
                rescue_thresh=RESCUE_THRESH
            )

            train_pool = list(set(labeled_pool + list(pseudo_labels.keys())))
            print(f"✨ Train Set Size: {len(train_pool)} (Labeled: {len(labeled_pool)} + Pseudo: {len(pseudo_labels)})")
            if len(train_pool) == 0: continue

            random.Random(SEED + epoch).shuffle(train_pool)
            full_len = (len(train_pool) // BATCH_SIZE) * BATCH_SIZE
            train_pool = train_pool[:full_len]

            bert_model_stage2.train()
            gat_model_stage2.train()
            total_loss = 0
            total_n = 0

            for i in range(0, len(train_pool), BATCH_SIZE):
                batch_gids = train_pool[i:i + BATCH_SIZE]

                # --- A. 准备 Weak View 数据 (原始) ---
                bert_samples = [bert_dataset_full[global_to_bert_idx[g]] for g in batch_gids]
                input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(bert_samples)
                if input_ids is None: continue
                token_type_ids = fix_token_type_shape(token_type_ids)
                input_ids, attn_mask, token_type_ids = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(
                    DEVICE)

                graphs = [gat_dataset_local[global_to_gat_idx[g]] for g in batch_gids]
                graph_batch = GraphBatch.from_data_list(graphs).to(DEVICE)

                targets = []
                for g in batch_gids:
                    if global_labels[g] != -100:
                        targets.append(global_labels[g])
                    elif g in pseudo_labels:
                        targets.append(pseudo_labels[g][0])
                    else:
                        targets.append(-100)
                targets_t = torch.tensor(targets, dtype=torch.long).to(DEVICE)
                graph_batch.y = targets_t.view(-1, 1)

                bert_opt.zero_grad()
                gat_opt_stage2.zero_grad()

                # --- B. Forward Weak View ---
                b_logits = get_logits(bert_model_stage2(input_ids, attn_mask, token_type_ids))
                g_logits = get_logits(gat_model_stage2(graph_batch.x, graph_batch.edge_index, graph_batch.batch))

                loss_sup_b = criterion(b_logits, targets_t)
                loss_sup_g = criterion(g_logits, targets_t)

                # --- C. Forward Strong View (强增强) - 使用路由包装器 ---
                # 1. BERT Strong View
                input_ids_s, attn_mask_s, token_type_ids_s = apply_strong_aug_bert(
                    input_ids, attn_mask, token_type_ids, mode=STRONG_AUG_MODE["bert"]
                )
                b_logits_strong = get_logits(bert_model_stage2(input_ids_s, attn_mask_s, token_type_ids_s))

                # 2. GAT Strong View
                graph_batch_strong = apply_strong_aug_gat(
                    graph_batch, mode=STRONG_AUG_MODE["gat"], drop_rate=STRONG_AUG_MODE["drop_rate"]
                )
                g_logits_strong = get_logits(gat_model_stage2(
                    graph_batch_strong.x, graph_batch_strong.edge_index, graph_batch_strong.batch
                ))

                # --- D. FixMatch Consistency Loss ---
                loss_fix_b = criterion(b_logits_strong, targets_t)
                loss_fix_g = criterion(g_logits_strong, targets_t)

                # --- E. Total Loss ---
                final_loss = (loss_sup_b + loss_sup_g) + LAMBDA_FIXMATCH * (loss_fix_b + loss_fix_g)


                final_loss.backward()
                bert_opt.step()
                gat_opt_stage2.step()
                total_loss += final_loss.item()
                total_n += 1

            print(f"Avg Loss (Sup + FixMatch): {total_loss / total_n:.4f}")

            # =========================================================
            # Evaluation 基于当前 alpha 软投票
            # =========================================================

            # 1. 仅在验证集评估（严格避免泄露测试集）- 传入 current_alpha
            val_res, _ = evaluate_models_dynamic(
                bert_model_stage2, gat_model_stage2, val_indices,
                global_labels, global_to_bert_idx, global_to_gat_idx,
                bert_dataset_full, gat_dataset_local, DEVICE, BATCH_SIZE,
                alpha=current_alpha  # ★ 动态传入
            )
            print(
                f"Validation -> BERT Acc: {val_res['bert_acc']:.4f} | GAT Acc: {val_res['gat_acc']:.4f} | Fusion Acc: {val_res['fusion_acc']:.4f} | Fusion Macro-F1: {val_res['fusion_macro_f1']:.4f}")

            # 2. 更新最佳成绩逻辑 (★ 根据验证集 Val 选拔)
            current_val_f1 = val_res['fusion_macro_f1']
            if current_val_f1 > best_val_macro_f1_this_alpha:
                best_val_macro_f1_this_alpha = current_val_f1
                best_epoch_this_alpha = epoch

                # 为当前 Alpha 单独保存一份最优模型权重
                torch.save(bert_model_stage2.state_dict(), os.path.join(ROOT, f"mix_bert_best_alpha.pt"))
                torch.save(gat_model_stage2.state_dict(), os.path.join(ROOT, f"mix_gat_best_alpha.pt"))
                print(
                    f">>> ⚡ Validation improved for Alpha={current_alpha:.1f} (Val Macro-F1={best_val_macro_f1_this_alpha:.4f})! Models saved.")

        # ======================================================
        # 当前 Alpha 循环结束：在其独立测试集上进行终极评价
        # ======================================================
        print(f"\n✅ Alpha = {current_alpha:.1f} 的完整训练已结束！")

        # 加载本轮跑出的最好模型
        bert_model_stage2.load_state_dict(torch.load(os.path.join(ROOT, f"mix_bert_best_alpha.pt")))
        gat_model_stage2.load_state_dict(torch.load(os.path.join(ROOT, f"mix_gat_best_alpha.pt")))
        bert_model_stage2.eval()
        gat_model_stage2.eval()

        test_res, _ = evaluate_models_dynamic(
            bert_model_stage2, gat_model_stage2, test_indices,
            global_labels, global_to_bert_idx, global_to_gat_idx,
            bert_dataset_full, gat_dataset_local, DEVICE, BATCH_SIZE,
            alpha=current_alpha  # ★ 动态传入
        )

        print(f"   ► 该参数下最佳 Epoch 出现于: {best_epoch_this_alpha}")
        print(f"   ► 验证集峰值 Macro-F1: {best_val_macro_f1_this_alpha:.4f}")
        print(
            f"   ► 对应的测试集指标 -> Acc: {test_res.get('fusion_acc', 0):.4f}, F1: {test_res.get('fusion_macro_f1', 0):.4f}")

        # === 比较并更新全局总冠军 ===
        if best_val_macro_f1_this_alpha > global_best_val_macro_f1:
            global_best_val_macro_f1 = best_val_macro_f1_this_alpha
            global_best_alpha = current_alpha
            global_best_test_metrics = test_res

    # ======================================================
    # ★ 所有 Alpha 步进结束后的最终总结
    # ======================================================
    print("\n" + "🌟" * 30)
    print("🎉 所有 Alpha (0.1 -> 0.9) 步进搜索完毕！完美收官！")
    print(
        f"👑 【全局最优 Alpha 参数】: {global_best_alpha:.1f} (即 BERT 权重为 {global_best_alpha:.1f}, GAT 权重为 {1.0 - global_best_alpha:.1f})")
    print(f"📈 【最佳验证集 Macro-F1】: {global_best_val_macro_f1:.4f}")

    print(f"\n📊 【该最优 Alpha 在独立测试集 (Test Set) 上的终极表现】:")
    print(f"【Fusion 全局联合指标】")
    print(f"   ► Accuracy  (总准确率)   : {global_best_test_metrics.get('fusion_acc', 0):.4f}")
    print(f"   ► Macro-F1  (宏平均F1)   : {global_best_test_metrics.get('fusion_macro_f1', 0):.4f}")
    print(f"【Fusion 真言论 / True Claims (Label 0)】")
    print(f"   ► Precision (精确率)     : {global_best_test_metrics.get('true_p', 0):.4f}")
    print(f"   ► Recall    (召回率)     : {global_best_test_metrics.get('true_r', 0):.4f}")
    print(f"   ► F1-Score  (F1值)       : {global_best_test_metrics.get('true_f1', 0):.4f}")
    print(f"【Fusion 假言论 / False Claims (Label 1)】")
    print(f"   ► Precision (精确率)     : {global_best_test_metrics.get('false_p', 0):.4f}")
    print(f"   ► Recall    (召回率)     : {global_best_test_metrics.get('false_r', 0):.4f}")
    print(f"   ► F1-Score  (F1值)       : {global_best_test_metrics.get('false_f1', 0):.4f}")
    print("🌟" * 30 + "\n")


if __name__ == "__main__":
    main()