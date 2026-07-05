import os
import json
import warnings
import torch
import pandas as pd
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.utils import dropout_adj


# ==============================================================================
# 1. 基础工具函数
# ==============================================================================

def get_logits(outputs):
    """安全地从模型输出中提取 logits"""
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


def fix_token_type_shape(ttids):
    ttids = ttids.long()
    if ttids.dim() == 3:
        if ttids.size(1) == 1:
            ttids = ttids.squeeze(1)
        elif ttids.size(2) == 1:
            ttids = ttids.squeeze(2)
        else:
            ttids = ttids[..., 0]
    return ttids


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

    if len(input_ids_list) == 0:
        return None, None, None, None

    return torch.stack(input_ids_list), torch.stack(attention_mask_list), torch.stack(
        token_type_ids_list), torch.tensor(labels_list, dtype=torch.long)


def build_common_alignment(json_root, graph_dataset):
    bert_files = sorted([f for f in os.listdir(json_root) if f.endswith(".json")])
    bert_ids = [os.path.splitext(f)[0] for f in bert_files]
    if hasattr(graph_dataset, 'processed_file_names'):
        gat_filenames = [os.path.splitext(fn)[0] for fn in graph_dataset.processed_file_names]
    else:
        gat_filenames = [getattr(graph_dataset[i], 'sample_id', str(i)) for i in range(len(graph_dataset))]
    common_ids = [sid for sid in bert_ids if sid in set(gat_filenames)]
    return common_ids, bert_ids, gat_filenames


# ==============================================================================
# 2. 强增强模块 (Strong Augmentation) - FixMatch 核心
# ==============================================================================

def strong_aug_bert_shuffle(input_ids, attention_mask, shuffle_prob=0.15):
    """
    BERT 强增强：随机打乱 (Partial Token Shuffling)
    """
    aug_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.size()

    for i in range(batch_size):
        # 1. 获取真实长度 (去掉 padding)
        valid_len = attention_mask[i].sum().item()
        if valid_len <= 3: continue  # 句子太短就不折腾了

        # 锁定目标区域：排除开头 [CLS] 和结尾 [SEP]
        start_idx = 1
        end_idx = valid_len - 1
        mid_len = end_idx - start_idx

        if mid_len <= 1: continue

        # 2. 计算要打乱多少个 token
        num_to_shuffle = max(1, int(mid_len * shuffle_prob))

        # 3. 随机选出 num_to_shuffle 个要“动刀”的位置
        shuffle_indices = torch.randperm(mid_len)[:num_to_shuffle]
        target_indices = start_idx + shuffle_indices

        # 4. 取出这些位置原本的值
        target_values = aug_input_ids[i, target_indices]

        # 5. 将取出的这些值内部再次打乱
        perm_for_values = torch.randperm(num_to_shuffle)
        shuffled_values = target_values[perm_for_values]

        # 6. 填回去
        aug_input_ids[i, target_indices] = shuffled_values

    return aug_input_ids

def strong_aug_gat_dropedge(edge_index, p=0.3):
    """
    GAT 强增强：随机剪枝 (DropEdge)
    """
    new_edge_index, _ = dropout_adj(edge_index, p=p, force_undirected=True)
    return new_edge_index

def apply_strong_aug_bert(input_ids, attn_mask, token_type_ids, mode="shuffle"):
    """BERT强增强路由"""
    if mode == "shuffle":
        return strong_aug_bert_shuffle(input_ids, attn_mask, token_type_ids)
    elif mode == "identity":
        return input_ids, attn_mask, token_type_ids  # 弱视图直接复用
    else:
        raise ValueError(f"Unknown BERT aug mode: {mode}")

def apply_strong_aug_gat(graph_batch, mode="dropedge", drop_rate=0.2):
    """GAT强增强路由"""
    if mode == "dropedge":
        return strong_aug_gat_dropedge(graph_batch, drop_rate=drop_rate)
    elif mode == "identity":
        return graph_batch  # 原始图直接复用
    else:
        raise ValueError(f"Unknown GAT aug mode: {mode}")

# ==============================================================================
# 3. 核心策略逻辑：共识筛选与评估
# ==============================================================================
def generate_pseudo_fixed_threshold(bert_model, gat_model, indices, g2b, g2g, bert_dataset, gat_dataset, device,
                                batch_size=16,
                                consensus_thresh=0.80, rescue_thresh=0.98):
    """
    【消融实验 G4 组：经典半监督范式 (FixMatch 多模态基线)】
    策略：线性融合 BERT 与 GAT 概率 (固定 α=0.5) → 单阈值过滤 (≥ consensus_thresh)
    注意：rescue_thresh 在此范式下不启用，仅保留接口兼容性
    """
    bert_model.eval()
    gat_model.eval()
    pseudo = {}
    stats = {"accepted": 0, "dropped": 0}

    # 经典范式通常采用等权融合，避免引入动态调优干扰
    alpha = 0.5

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # BERT Prep (保持原逻辑)
            bert_samples = [bert_dataset[g2b[idx]] for idx in batch_idx]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(bert_samples)
            if input_ids is None: continue
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            # GAT Prep (保持原逻辑)
            graphs = [gat_dataset[g2g[idx]] for idx in batch_idx]
            if len(graphs) == 0: continue
            graph_batch = GraphBatch.from_data_list(graphs).to(device)

            # Inference (保持原逻辑)
            b_out = bert_model(input_ids, attn_mask, token_type_ids)
            g_out = gat_model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

            b_probs = torch.softmax(get_logits(b_out), dim=1).cpu()
            g_probs = torch.softmax(get_logits(g_out), dim=1).cpu()

            # ★ 核心修改：先融合概率，后应用单阈值决策
            fused_probs = alpha * b_probs + (1 - alpha) * g_probs
            fused_conf, fused_pred = fused_probs.max(dim=1)

            for gid, conf, pred in zip(batch_idx, fused_conf.tolist(), fused_pred.tolist()):
                # 经典 FixMatch：仅当融合置信度 ≥ 阈值时采纳
                if conf >= consensus_thresh:
                    # 保持原返回格式: (label, confidence_score, tag) 兼容下游解析
                    pseudo[gid] = (int(pred), float(conf), 'classical_fusion')
                    stats["accepted"] += 1
                else:
                    stats["dropped"] += 1

    print(f"  [Stats] Accepted: {stats['accepted']}, Dropped: {stats['dropped']}")
    return pseudo


def generate_pseudo_from_fusion(bert_model, gat_model, indices, g2b, g2g, bert_dataset, gat_dataset, device,
                                batch_size=16,
                                consensus_thresh=0.80,
                                rescue_thresh=0.98):
    """
    Consensus + Expert Rescue 策略生成伪标签
    """
    bert_model.eval()
    gat_model.eval()
    pseudo = {}
    stats = {"consensus": 0, "bert_rescue": 0, "gat_rescue": 0, "dropped": 0}

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # BERT Prep
            bert_samples = [bert_dataset[g2b[idx]] for idx in batch_idx]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(bert_samples)
            if input_ids is None: continue
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            # GAT Prep
            graphs = [gat_dataset[g2g[idx]] for idx in batch_idx]
            if len(graphs) == 0: continue
            graph_batch = GraphBatch.from_data_list(graphs).to(device)

            # Inference
            b_out = bert_model(input_ids, attn_mask, token_type_ids)
            g_out = gat_model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

            b_probs = torch.softmax(get_logits(b_out), dim=1).cpu()
            g_probs = torch.softmax(get_logits(g_out), dim=1).cpu()

            b_maxp, b_pred = b_probs.max(dim=1)
            g_maxp, g_pred = g_probs.max(dim=1)

            for gid, bp, bpr, gp, gpr in zip(batch_idx,
                                             b_maxp.tolist(), b_pred.tolist(),
                                             g_maxp.tolist(), g_pred.tolist()):
                if bpr == gpr:
                    # 更改逻辑：直接判断两个模型的置信度是否都独立达到了阈值
                    if bp >= consensus_thresh and gp >= consensus_thresh:
                        # 达成共识时的分数保存为两者的平均值（也可以保存其中一个，这里选用平均以作记录）
                        pseudo[gid] = (int(bpr), (bp + gp) / 2.0, 'consensus')
                        stats["consensus"] += 1
                    else:
                        stats["dropped"] += 1
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
                            gat_dataset_local, device, batch_size=16, alpha=0.5):
    """
    独立评估 BERT 和 GAT 模型的准确率，并引入基于传入 alpha 参数的软投票机制。
    """
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix  # 我加的

    bert_model_local.eval()
    gat_model_local.eval()
    all_bert_logits = []
    all_gat_logits = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # BERT 推理
            samples = [bert_dataset_full[g2b[idx]] for idx in batch_idx]
            input_ids, attn_mask, token_type_ids, _ = collate_bert_batch_from_samples(samples)
            if input_ids is None: continue
            token_type_ids = fix_token_type_shape(token_type_ids)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            b_out = bert_model_local(input_ids, attn_mask, token_type_ids)

            # GAT 推理
            graphs = [gat_dataset_local[g2g[idx]] for idx in batch_idx]
            if len(graphs) == 0: continue
            graph_batch = GraphBatch.from_data_list(graphs).to(device)
            g_out = gat_model_local(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

            true = torch.tensor([global_labels[idx] for idx in batch_idx], dtype=torch.long)

            all_bert_logits.append(get_logits(b_out).cpu())
            all_gat_logits.append(get_logits(g_out).cpu())
            all_labels.append(true)

    if len(all_labels) == 0: return {'bert': 0.0, 'gat': 0.0, 'fusion': 0.0}, 0

    bert_all = torch.cat(all_bert_logits, dim=0)
    gat_all = torch.cat(all_gat_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    # 1. 分别计算两个模型的独立准确率 (仅作为 Baseline 参考)
    bert_acc = (bert_all.argmax(dim=1) == labels_all).float().mean().item()
    gat_acc = (gat_all.argmax(dim=1) == labels_all).float().mean().item()

    # =========================================================
    # ★ 核心操作：仅在输出阶段进行的 Soft Voting 融合 ★
    # =========================================================
    # 将 logits 转换为概率分布
    bert_probs = F.softmax(bert_all, dim=1)
    gat_probs = F.softmax(gat_all, dim=1)

    # 使用传入的 alpha 进行融合
    fusion_probs = alpha * bert_probs + (1.0 - alpha) * gat_probs
    final_preds = fusion_probs.argmax(dim=1).numpy()

    # ★ 新增：计算所有需要的豪华指标
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    fusion_acc = accuracy_score(labels_all, final_preds)
    fusion_macro_f1 = f1_score(labels_all, final_preds, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(labels_all, final_preds, labels=[0, 1],
                                                                    zero_division=0)

    # ===================== ✅ 关键在这里：我加的 TP/FP/FN/TN =====================
    tn, fp, fn, tp = confusion_matrix(labels_all, final_preds).ravel()
    # ===========================================================================

    # 将所有的细粒度指标打包返回
    return {
        'bert_acc': bert_acc,
        'gat_acc': gat_acc,
        'fusion_acc': fusion_acc,
        'fusion_macro_f1': fusion_macro_f1,
        'true_p': p_class[0], 'true_r': r_class[0], 'true_f1': f1_class[0],
        'false_p': p_class[1], 'false_r': r_class[1], 'false_f1': f1_class[1],
        # 👇 下面四个是我新增返回的，直接就能用
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }, len(labels_all)

# ==============================================================================
# 4. ???????
# ==============================================================================
def evaluate_pseudo_accuracy(pseudo_labels, global_labels):
    """
    ??????????????????????????????

    ??
    ----
    pseudo_labels : dict {global_index: (pred_label, confidence, tag), ...}
    global_labels : list [label_or_-100, ...]  ? common_ids ??

    ??
    ----
    dict : {
        'accuracy':      ?????,
        'correct':       ???,
        'total':         ?????,
        'per_class':     {0: {'correct': N, 'total': M, 'accuracy': ...}, 1: {...}},
        'source_dist':   {tag: count, ...}
    }
    """
    from collections import Counter

    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    source_dist = Counter()

    for gid, (pred, conf, tag) in pseudo_labels.items():
        truth = global_labels[gid]
        if truth == -100:
            continue
        total += 1
        source_dist[tag] += 1
        class_total[truth] += 1
        if pred == truth:
            correct += 1
            class_correct[truth] += 1

    per_class = {}
    for cls in sorted(set(list(class_total.keys()) + list(class_correct.keys()))):
        c = class_correct.get(cls, 0)
        t = class_total.get(cls, 0)
        per_class[cls] = {
            'correct': c,
            'total': t,
            'accuracy': c / t if t > 0 else 0.0
        }

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'source_dist': dict(source_dist)
    }

def print_pseudo_evaluation(pseudo_labels, global_labels, prefix="  [Pseudo Quality]"):
    """
    ?? evaluate_pseudo_accuracy??????????
    """
    stats = evaluate_pseudo_accuracy(pseudo_labels, global_labels)
    if stats['total'] == 0:
        print(f"{prefix} No pseudo labels with ground truth available.")
        return stats

    print(f"{prefix} === Pseudo-Label Quality (vs Ground Truth) ===")
    print(f"{prefix} Accuracy : {stats['correct']}/{stats['total']} = {stats['accuracy']*100:.2f}%")

    for cls, info in sorted(stats['per_class'].items()):
        label_name = "Non-Rumor" if cls == 0 else "Rumor"
        print(f"{prefix}   Class {cls} ({label_name}): {info['correct']}/{info['total']} = {info['accuracy']*100:.2f}%")

    if stats['source_dist']:
        dist_str = ", ".join(f"{k}: {v}" for k, v in stats['source_dist'].items())
        print(f"{prefix} Source distribution: {dist_str}")

    print(f"{prefix} {'=' * 40}")
    return stats
