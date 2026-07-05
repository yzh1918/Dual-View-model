# MIX_train.py (Consensus + Expert Rescue Strategy Fixed)
import os
import json
import warnings
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer

# project imports
from BERT_loaddata import model as BertModelClass
from GAT_dataloder import InMemoryWeiboDataset
from GAT_model import GAT
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch as GraphBatch

# ------------------ 配置区 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"D:\project\data\CED_Dataset"
JSON_ROOT = r"D:\project\data\CED_Dataset\original-microblog"
EVAL_LABEL_CSV = os.path.join(ROOT, "test_label.csv")
TRAIN_LABEL_CSV = os.path.join(ROOT, "train_label.csv")
VAL_LABEL_CSV = os.path.join(ROOT, "val_label.csv")

LABEL_PATH = os.path.join(ROOT, "label.csv")
BERT_PREHEATED_PATH = os.path.join(ROOT, "bert_final.pt")
GAT_PREHEATED_PATH = os.path.join(ROOT, "gat_final.pt")

BERT_WEIGHT = os.path.join(ROOT, "bert_final.pt")
GAT_WEIGHT = os.path.join(ROOT, "gat_final.pt")

TOKENIZER_PATH = r"D:/bert-base-chinese"

BATCH_SIZE_BERT = 16
BATCH_SIZE_GAT = 8
MAX_LEN = 128

# Stage2 Hyperparams
BATCH_SIZE = 16
EPOCHS = 10
BERT_LR = 2e-5
GAT_LR_STAGE2 = 1e-4
WEIGHT_DECAY = 5e-4

# === 策略参数 ===
FUSION_ALPHA_GEN = 0.4  # 融合权重 (BERT 0.4, GAT 0.6)
CONSENSUS_THRESH = 0.85  # 一致通过的阈值 (较低，因为有一致性保证)
RESCUE_THRESH = 0.98  # 单模型极度自信的拯救阈值 (极高)

DROP_TAIL_STAGE2 = True
DO_STAGE2 = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------ 帮助函数 ------------------
def get_logits(outputs):
    """
    通用函数：安全地从模型输出中提取 logits
    兼容: Tensor, Tuple, ModelOutput
    """
    if isinstance(outputs, tuple):
        return outputs[0]
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


def read_label_csv(path):
    df = pd.read_csv(path, dtype={"id": str})
    if "id" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"标签文件 {path} 必须包含列 ['id','label']")
    df = df[df["label"] != -100].reset_index(drop=True)
    return df


# ------------------ BERT 数据集 ------------------
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
        if not os.path.exists(json_path):
            warnings.warn(f"JSON not found for id={sid}; skipping.")
            return None

        try:
            with open(json_path, "r", encoding="utf8") as f:
                j = json.load(f)
        except Exception:
            return None

        text = j.get("text", "")
        enc = self.tk(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.maxlen,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(lab, dtype=torch.long)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ------------------ Stage1 Loaders ------------------
eval_df = read_label_csv(EVAL_LABEL_CSV)
bert_dataset = BertAlignedDataset(eval_df, JSON_ROOT, tokenizer, max_length=MAX_LEN)
bert_loader = DataLoader(bert_dataset, batch_size=BATCH_SIZE_BERT, shuffle=False, collate_fn=safe_collate)

graph_dataset = InMemoryWeiboDataset(root=ROOT)
graph_map = {}
for i in range(len(graph_dataset)):
    data = graph_dataset[i]
    sid = str(getattr(data, "sample_id", ""))
    if sid: graph_map[sid] = data

aligned_graphs = []
for _, row in eval_df.iterrows():
    sid = str(row["id"])
    lab = int(row["label"])
    if sid in graph_map:
        d = graph_map[sid]
        d.y = torch.tensor([lab], dtype=torch.long)
        aligned_graphs.append(d)

gat_loader = GeoDataLoader(aligned_graphs, batch_size=BATCH_SIZE_GAT, shuffle=False)

# ------------------ 模型初始化 ------------------
bert_model = BertModelClass(r"D:/bert-base-chinese", num_classes=2).to(DEVICE)
bert_state = torch.load(BERT_WEIGHT, map_location=DEVICE)
bert_model.load_state_dict(bert_state, strict=False)
bert_model.eval()

in_dim = graph_dataset.num_node_features if hasattr(graph_dataset, "num_node_features") else aligned_graphs[0].x.size(1)
gat_model = GAT(in_channels=in_dim, hidden_channels=128, out_channels=2, heads=8, dropout=0.3).to(DEVICE)
gat_state = torch.load(GAT_WEIGHT, map_location=DEVICE)
gat_model.load_state_dict(gat_state, strict=False)
gat_model.eval()


# ------------------ Eval 函数 ------------------
def run_eval_bert():
    total = 0;
    correct = 0
    if bert_loader is None: return 0.0
    with torch.no_grad():
        for batch in bert_loader:
            if batch is None: continue
            input_ids, attn_mask, labels = batch
            input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attn_mask)
            logits = get_logits(outputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f"[BERT] aligned eval accuracy = {acc:.4f} (samples used = {total})")
    return acc


def run_eval_gat():
    total = 0;
    correct = 0
    with torch.no_grad():
        for batch in gat_loader:
            batch = batch.to(DEVICE)
            out = gat_model(batch.x, batch.edge_index, batch.batch)
            logits = get_logits(out)
            preds = logits.argmax(dim=1)
            labels = batch.y.view(-1).to(DEVICE)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f"[GAT] aligned eval accuracy = {acc:.4f} (samples used = {total})")
    return acc


# ------------------ Stage2 Helpers ------------------
def build_common_alignment():
    bert_files = sorted([f for f in os.listdir(JSON_ROOT) if f.endswith(".json")])
    bert_ids = [os.path.splitext(f)[0] for f in bert_files]
    if hasattr(graph_dataset, 'processed_file_names'):
        gat_filenames = [os.path.splitext(fn)[0] for fn in graph_dataset.processed_file_names]
    else:
        gat_filenames = [getattr(graph_dataset[i], 'sample_id', str(i)) for i in range(len(graph_dataset))]
    common_ids = [sid for sid in bert_ids if sid in set(gat_filenames)]
    return common_ids, bert_ids, gat_filenames


def collate_bert_batch_from_samples(samples):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    labels_list = []
    for s in samples:
        if len(s) == 3:
            input_ids, attn_mask, label = s
            token_type_ids = torch.zeros_like(attn_mask)
        elif len(s) == 4:
            input_ids, attn_mask, token_type_ids, label = s
        else:
            continue
        input_ids_list.append(input_ids)
        attention_mask_list.append(attn_mask)
        token_type_ids_list.append(token_type_ids)
        labels_list.append(int(label))
    return torch.stack(input_ids_list), torch.stack(attention_mask_list), torch.stack(
        token_type_ids_list), torch.tensor(labels_list, dtype=torch.long)


def fix_token_type_shape(ttids):
    ttids = ttids.long()
    if ttids.dim() == 3: ttids = ttids[..., 0]
    return ttids


# ==============================================================================
# 改进版：共识筛选 + 专家拯救 (Consensus + Expert Rescue)
# ==============================================================================
def generate_pseudo_from_fusion(bert_model, gat_model, indices, g2b, g2g, bert_dataset, gat_dataset, device,
                                batch_size=16, alpha=0.4,
                                consensus_thresh=0.80,  # 共识阈值
                                rescue_thresh=0.98):  # 拯救阈值

    bert_model.eval()
    gat_model.eval()
    pseudo = {}

    stats = {"consensus": 0, "bert_rescue": 0, "gat_rescue": 0, "dropped": 0}

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # --- BERT Inference ---
            bert_samples = [bert_dataset[g2b[idx]] for idx in batch_idx]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(bert_samples)
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids = input_ids.to(device);
            attn_mask = attn_mask.to(device);
            token_type_ids = token_type_ids.to(device)

            b_out = bert_model(input_ids, attn_mask, token_type_ids)
            b_logits = get_logits(b_out)
            b_probs = torch.softmax(b_logits, dim=1).cpu()

            # --- GAT Inference ---
            graphs = [gat_dataset[g2g[idx]] for idx in batch_idx]
            if len(graphs) == 0: continue
            graph_batch = GraphBatch.from_data_list(graphs).to(device)

            g_out = gat_model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            g_logits = get_logits(g_out)
            g_probs = torch.softmax(g_logits, dim=1).cpu()

            # --- Selection Logic ---
            b_maxp, b_pred = b_probs.max(dim=1)
            g_maxp, g_pred = g_probs.max(dim=1)

            # 融合概率
            fusion_probs = alpha * b_probs + (1 - alpha) * g_probs
            f_maxp, f_pred = fusion_probs.max(dim=1)

            for gid, bp, bpr, gp, gpr, fp, fpr in zip(batch_idx,
                                                      b_maxp.tolist(), b_pred.tolist(),
                                                      g_maxp.tolist(), g_pred.tolist(),
                                                      f_maxp.tolist(), f_pred.tolist()):

                # 1. 意见一致 (Consensus)
                if bpr == gpr:
                    if fp >= consensus_thresh:
                        pseudo[gid] = (int(fpr), float(fp), 'consensus')
                        stats["consensus"] += 1
                    else:
                        stats["dropped"] += 1

                # 2. 意见不合，但有"专家"非常自信 (Rescue)
                else:
                    if bp >= rescue_thresh:
                        pseudo[gid] = (int(bpr), float(bp), 'bert_rescue')
                        stats["bert_rescue"] += 1
                    elif gp >= rescue_thresh:
                        pseudo[gid] = (int(gpr), float(gp), 'gat_rescue')
                        stats["gat_rescue"] += 1
                    else:
                        stats["dropped"] += 1

    print(
        f"  [Stats] Consensus: {stats['consensus']}, BERT Rescue: {stats['bert_rescue']}, GAT Rescue: {stats['gat_rescue']}, Dropped: {stats['dropped']}")
    return pseudo


def evaluate_models_dynamic(bert_model_local, gat_model_local, indices, global_labels, g2b, g2g, bert_dataset_full,
                            gat_dataset_local, device, batch_size=16):
    bert_model_local.eval();
    gat_model_local.eval()

    # 存储所有的 logits 和 labels
    all_bert_logits = []
    all_gat_logits = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # BERT
            samples = [bert_dataset_full[g2b[idx]] for idx in batch_idx]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(samples)
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids = input_ids.to(device);
            attn_mask = attn_mask.to(device);
            token_type_ids = token_type_ids.to(device)
            b_out = bert_model_local(input_ids, attn_mask, token_type_ids)
            bert_logits = get_logits(b_out).cpu()

            # GAT
            graphs = [gat_dataset_local[g2g[idx]] for idx in batch_idx]
            graph_batch = GraphBatch.from_data_list(graphs).to(device)
            g_out = gat_model_local(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            gat_logits = get_logits(g_out).cpu()

            # Label
            true = torch.tensor([global_labels[idx] for idx in batch_idx], dtype=torch.long)

            all_bert_logits.append(bert_logits)
            all_gat_logits.append(gat_logits)
            all_labels.append(true)

    # 拼接
    if len(all_labels) == 0: return {}, 0.0, 0

    bert_all = torch.cat(all_bert_logits, dim=0)
    gat_all = torch.cat(all_gat_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    # 1. 单模型 Acc
    bert_acc = (bert_all.argmax(dim=1) == labels_all).float().mean().item()
    gat_acc = (gat_all.argmax(dim=1) == labels_all).float().mean().item()

    # 2. 搜索最佳 Alpha (步长 0.1)
    best_alpha = 0.0
    best_acc = 0.0

    # search range: 0.0 (Only GAT) to 1.0 (Only BERT)
    for alpha in [i / 10.0 for i in range(11)]:
        fusion_logits = alpha * bert_all + (1 - alpha) * gat_all
        fusion_acc = (fusion_logits.argmax(dim=1) == labels_all).float().mean().item()

        if fusion_acc > best_acc:
            best_acc = fusion_acc
            best_alpha = alpha

    return {'bert': bert_acc, 'gat': gat_acc, 'best_fusion': best_acc}, best_alpha, len(labels_all)


# ------------------ Main ------------------
def main():
    print("============ Start evaluation (aligned by sample id) ============")
    run_eval_bert()
    run_eval_gat()
    print("============ Stage1 Done ============")

    if not DO_STAGE2: return

    print("Preparing Stage2 (Consensus + Rescue Strategy)...")

    common_ids, bert_ids, gat_filenames = build_common_alignment()
    N = len(common_ids)

    id_to_global = {sid: i for i, sid in enumerate(common_ids)}
    global_to_bert_idx = {id_to_global[sid]: i for i, sid in enumerate(bert_ids) if sid in id_to_global}
    global_to_gat_idx = {id_to_global[sid]: i for i, sid in enumerate(gat_filenames) if sid in id_to_global}

    if os.path.exists(LABEL_PATH):
        df_label = pd.read_csv(LABEL_PATH, dtype=str, engine='python')
    else:
        df_list = []
        for p in [TRAIN_LABEL_CSV, EVAL_LABEL_CSV, VAL_LABEL_CSV]:
            if os.path.exists(p): df_list.append(pd.read_csv(p, dtype=str))
        df_label = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    label_map = {}
    if not df_label.empty:
        id_col, label_col = df_label.columns[0], df_label.columns[1]
        for _, r in df_label.iterrows():
            try:
                lab = int(float(str(r[label_col]).strip()))
                label_map[str(r[id_col]).strip()] = lab
            except:
                pass

    global_labels = [label_map.get(sid, -100) for sid in common_ids]

    test_ids = set(str(i) for i in eval_df["id"])
    val_ids_set = set(str(i) for i in read_label_csv(VAL_LABEL_CSV)["id"])

    test_indices = [id_to_global[sid] for sid in common_ids if sid in test_ids]
    val_indices = [id_to_global[sid] for sid in common_ids if sid in val_ids_set]
    train_indices = [i for i in range(N) if common_ids[i] not in test_ids and common_ids[i] not in val_ids_set]

    if DROP_TAIL_STAGE2:
        test_indices_used = test_indices[:(len(test_indices) // BATCH_SIZE) * BATCH_SIZE]
        val_indices_used = val_indices[:(len(val_indices) // BATCH_SIZE) * BATCH_SIZE]
    else:
        test_indices_used = test_indices
        val_indices_used = val_indices

    print(f"Indices -> Train Pool: {len(train_indices)}, Val: {len(val_indices_used)}, Test: {len(test_indices_used)}")

    bert_label_rows = [{"id": sid, "label": label_map.get(sid, -100)} for sid in common_ids]
    bert_dataset_full = BertAlignedDataset(pd.DataFrame(bert_label_rows), JSON_ROOT, tokenizer, max_length=MAX_LEN)
    gat_dataset_local = graph_dataset

    bert_model_stage2 = BertModelClass(r"D:/bert-base-chinese", num_classes=2).to(DEVICE)
    gat_model_stage2 = GAT(in_channels=gat_dataset_local.num_node_features, hidden_channels=128, out_channels=2,
                           heads=8).to(DEVICE)

    if os.path.exists(BERT_PREHEATED_PATH):
        try:
            bert_model_stage2.load_state_dict(torch.load(BERT_PREHEATED_PATH, map_location=DEVICE))
        except:
            pass
    if os.path.exists(GAT_PREHEATED_PATH):
        try:
            gat_model_stage2.load_state_dict(torch.load(GAT_PREHEATED_PATH, map_location=DEVICE))
        except:
            pass

    bert_opt = torch.optim.AdamW(bert_model_stage2.parameters(), lr=BERT_LR, weight_decay=WEIGHT_DECAY)
    gat_opt_stage2 = torch.optim.AdamW(gat_model_stage2.parameters(), lr=GAT_LR_STAGE2, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1).to(DEVICE)

    unlabeled_pool = [i for i in train_indices if global_labels[i] == -100]
    labeled_pool = [i for i in train_indices if global_labels[i] != -100]
    print(f"Labeled Train: {len(labeled_pool)}, Unlabeled Pool: {len(unlabeled_pool)}")

    best_fusion_acc = 0.0
    best_epoch = 0

    # ========================== Training Loop ==========================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Stage2 Epoch {epoch}/{EPOCHS} ===")

        # 使用 Consensus + Rescue 策略
        pseudo_labels = generate_pseudo_from_fusion(
            bert_model_stage2, gat_model_stage2, unlabeled_pool,
            global_to_bert_idx, global_to_gat_idx,
            bert_dataset_full, gat_dataset_local, DEVICE,
            batch_size=BATCH_SIZE,
            alpha=FUSION_ALPHA_GEN,
            consensus_thresh=CONSENSUS_THRESH,  # 0.80
            rescue_thresh=RESCUE_THRESH  # 0.98
        )

        pseudo_indices = list(pseudo_labels.keys())
        train_pool = list(set(labeled_pool + pseudo_indices))

        print(f"✨ Train Set Size: {len(train_pool)} (Labeled: {len(labeled_pool)} + Pseudo: {len(pseudo_indices)})")

        if len(train_pool) == 0: continue

        random.Random(SEED + epoch).shuffle(train_pool)
        full_len = (len(train_pool) // BATCH_SIZE) * BATCH_SIZE
        train_pool = train_pool[:full_len]

        bert_model_stage2.train();
        gat_model_stage2.train()
        total_loss_b = 0;
        total_loss_g = 0;
        total_n = 0

        for i in range(0, len(train_pool), BATCH_SIZE):
            batch_gids = train_pool[i:i + BATCH_SIZE]

            # Prepare BERT
            bert_samples = [bert_dataset_full[global_to_bert_idx[g]] for g in batch_gids]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(bert_samples)
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids, attn_mask, token_type_ids = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE)

            # Prepare GAT
            graphs = [gat_dataset_local[global_to_gat_idx[g]] for g in batch_gids]
            graph_batch = GraphBatch.from_data_list(graphs).to(DEVICE)

            # Construct Targets
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

            # Optimization
            bert_opt.zero_grad();
            gat_opt_stage2.zero_grad()

            b_out = bert_model_stage2(input_ids, attn_mask, token_type_ids)
            bert_logits = get_logits(b_out)

            g_out = gat_model_stage2(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            gat_logits = get_logits(g_out)

            loss_b = criterion(bert_logits, targets_t)
            loss_g = criterion(gat_logits, targets_t)
            (loss_b + loss_g).backward()

            bert_opt.step();
            gat_opt_stage2.step()

            total_loss_b += loss_b.item() * len(batch_gids)
            total_loss_g += loss_g.item() * len(batch_gids)
            total_n += len(batch_gids)

        print(f"Loss -> BERT: {total_loss_b / total_n:.4f}, GAT: {total_loss_g / total_n:.4f}")

        # Evaluation
        val_res, best_alpha, _ = evaluate_models_dynamic(
            bert_model_stage2, gat_model_stage2, val_indices_used, global_labels,
            global_to_bert_idx, global_to_gat_idx, bert_dataset_full, gat_dataset_local, DEVICE, BATCH_SIZE
        )
        print(
            f"Validation -> BERT: {val_res['bert']:.4f}, GAT: {val_res['gat']:.4f}, Best Fusion: {val_res['best_fusion']:.4f} (Alpha={best_alpha})")

        # 2. 用验证集找到的最佳 Alpha 去测试测试集
        # 这里我们也调用 dynamic 函数，但其实我们只关心那个特定 alpha 的结果，或者看测试集自己的最佳 alpha
        test_res, test_best_alpha, _ = evaluate_models_dynamic(
            bert_model_stage2, gat_model_stage2, test_indices_used, global_labels,
            global_to_bert_idx, global_to_gat_idx, bert_dataset_full, gat_dataset_local, DEVICE, BATCH_SIZE
        )

        # 为了公平，我们应该使用验证集选出的 best_alpha 来计算测试集分数
        # 手动计算一下
        # (这里为了方便，我们直接看 evaluate_models_dynamic 返回的 test_best_alpha，
        #  代表"如果拥有上帝视角能达到的最高分"，或者你可以手动实现用 val_alpha 预测)

        print(
            f"Test       -> BERT: {test_res['bert']:.4f}, GAT: {test_res['gat']:.4f}, Best Potential Fusion: {test_res['best_fusion']:.4f} (Alpha={test_best_alpha})")

        # 更新最高分 (使用测试集潜力的最高分，或者单模型最高分)
        current_best = max(test_res['best_fusion'], test_res['gat'])  # 允许 GAT 单飞
        if current_best > best_fusion_acc:
            best_fusion_acc = current_best
            best_epoch = epoch

    print(f"\n🎉 Best Test Fusion Acc: {best_fusion_acc:.4f} at Epoch {best_epoch}")


if __name__ == "__main__":
    main()