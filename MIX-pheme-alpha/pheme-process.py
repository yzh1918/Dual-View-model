import os
import json
import torch
import pandas as pd
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel, BertConfig
from datetime import datetime
from tqdm import tqdm

# ----------------- USER CONFIG -----------------
# PHEME 数据集根目录
PHEME_DIR = r"D:\project\data\PHEME_dataset"

# PT文件输出目录（每条源推文/Thread一个 .pt 文件）
OUT_DIR = os.path.join(PHEME_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# 全局标签 CSV 输出路径
CSV_OUT_PATH = os.path.join(PHEME_DIR, "full_label.csv")

# 英文 BERT 路径
BERT_FOLDER = r"D:\bert-base-uncased"

# BERT settings
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ------------------------------------------------

def get_label_from_annotation(annotation_path):
    """
    解析 annotation.json 并返回二分类标签。
    Rumor (false) -> 1, Non-rumor (true) -> 0
    返回 None 表示未经验证 (unverified) 或无效样本，应当丢弃。
    """
    if not os.path.exists(annotation_path):
        return None

    with open(annotation_path, 'r', encoding='utf-8') as f:
        try:
            annotation = json.load(f)
        except:
            return None

    if 'misinformation' in annotation and 'true' in annotation:
        if int(annotation['misinformation']) == 0 and int(annotation['true']) == 0:
            return None
        elif int(annotation['misinformation']) == 0 and int(annotation['true']) == 1:
            return 0  # true -> non-rumor (0)
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 0:
            return 1  # false -> rumor (1)
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 1:
            return None

    elif 'misinformation' in annotation and 'true' not in annotation:
        if int(annotation['misinformation']) == 0:
            return None
        elif int(annotation['misinformation']) == 1:
            return 1

    return None


def parse_twitter_timestamp(time_str):
    if not time_str:
        return None
    try:
        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
        return int(dt.timestamp())
    except Exception:
        try:
            dt = datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
            return int(dt.timestamp())
        except:
            return int(datetime.now().timestamp())


def time_features_from_ts(ts):
    dt = datetime.fromtimestamp(ts)
    hour = dt.hour / 23.0
    weekday = dt.weekday() / 6.0
    is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
    return [hour, weekday, float(is_weekend)]


class BertEncoder:
    def __init__(self, bert_folder, device):
        print("Loading BERT tokenizer & model (English)...")
        self.device = device
        self.config = BertConfig.from_pretrained(bert_folder, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_folder, local_files_only=True)
        self.model = BertModel.from_pretrained(bert_folder, config=self.config, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size
        print(f"BERT loaded. Hidden size: {self.dim}")

    def encode_single(self, text):
        if (text is None) or (not str(text).strip()):
            text = "[EMPTY]"
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return cls


def build_graph_for_thread(thread_dir, thread_id, label, bert_encoder):
    source_dir = os.path.join(thread_dir, "source-tweets")
    reactions_dir = os.path.join(thread_dir, "reactions")

    source_file = os.path.join(source_dir, f"{thread_id}.json")
    if not os.path.exists(source_file):
        source_file = os.path.join(source_dir, os.listdir(source_dir)[0])

    with open(source_file, 'r', encoding='utf-8') as f:
        orig = json.load(f)

    root_text = orig.get("text", "")
    root_time_str = orig.get("created_at", "")
    root_ts = parse_twitter_timestamp(root_time_str)
    root_time_feat = time_features_from_ts(root_ts) if root_ts is not None else [0.0, 0.0, 0.0]

    all_nodes = []
    all_nodes.append({
        "mid": str(orig.get("id_str", thread_id)),
        "text": root_text,
        "ts": root_ts,
        "time_feat": root_time_feat,
        "is_root": True
    })

    repost_items = []
    if os.path.exists(reactions_dir):
        for fname in os.listdir(reactions_dir):
            if not fname.endswith('.json'):
                continue
            react_file = os.path.join(reactions_dir, fname)
            try:
                with open(react_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                    ts = parse_twitter_timestamp(item.get("created_at", ""))
                    repost_items.append((ts, item))
            except Exception:
                continue

    repost_items.sort(key=lambda x: (x[0] if x[0] is not None else 0))

    for ts, item in repost_items:
        text = item.get("text", "")
        time_feat = time_features_from_ts(ts) if ts is not None else [0.0, 0.0, 0.0]
        all_nodes.append({
            "mid": str(item.get("id_str")),
            "text": text,
            "ts": ts,
            "time_feat": time_feat,
            "is_root": False,
            "raw_item": item
        })

    mid2idx = {node["mid"]: idx for idx, node in enumerate(all_nodes)}

    edges = []
    for idx, node in enumerate(all_nodes):
        if node["is_root"]:
            continue
        raw = node.get("raw_item", {})
        parent_mid = str(raw.get("in_reply_to_status_id_str", ""))
        p_idx = mid2idx.get(parent_mid, 0)

        edges.append([p_idx, idx])
        edges.append([idx, p_idx])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    node_feats = []
    for node in all_nodes:
        cls_feat = bert_encoder.encode_single(node.get("text", ""))
        time_feat = torch.tensor(node.get("time_feat", [0.0, 0.0, 0.0]), dtype=torch.float32)
        feat = torch.cat([cls_feat, time_feat], dim=0)
        node_feats.append(feat)

    x = torch.stack(node_feats, dim=0)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    data.sample_id = thread_id
    data.num_nodes = x.size(0)

    return data


def main():
    bert_enc = BertEncoder(BERT_FOLDER, DEVICE)
    topics = [d for d in os.listdir(PHEME_DIR) if os.path.isdir(os.path.join(PHEME_DIR, d)) and d != "processed"]

    processed_count = 0
    skipped_count = 0

    # 收集 CSV 数据的列表
    csv_data = []

    print(f"Found topics: {topics}")

    for topic in topics:
        topic_dir = os.path.join(PHEME_DIR, topic)
        for category in ['rumours', 'non-rumours']:
            cat_dir = os.path.join(topic_dir, category)
            if not os.path.exists(cat_dir):
                continue

            threads = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]

            for thread_id in tqdm(threads, desc=f"Processing {topic}/{category}"):
                thread_dir = os.path.join(cat_dir, thread_id)
                annotation_path = os.path.join(thread_dir, "annotation.json")

                label = get_label_from_annotation(annotation_path)
                if label is None:
                    skipped_count += 1
                    continue

                try:
                    data = build_graph_for_thread(thread_dir, thread_id, label, bert_enc)
                    out_path = os.path.join(OUT_DIR, f"{thread_id}.pt")
                    torch.save(data, out_path)

                    # 将成功的样本记录到列表中，准备写入 CSV
                    csv_data.append({'id': thread_id, 'label': label})
                    processed_count += 1

                except Exception as e:
                    print(f"\nError processing thread {thread_id}: {e}")
                    skipped_count += 1

    # 保存 CSV 文件
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(CSV_OUT_PATH, index=False)
        print(f"\n✅ Successfully saved full label CSV to: {CSV_OUT_PATH}")
    else:
        print("\n⚠️ No valid samples processed. CSV not generated.")

    print(f"✅ Done! Saved {processed_count} graphs to: {OUT_DIR}")
    print(f"⏭ Skipped (Unverified or errors): {skipped_count}")


if __name__ == "__main__":
    main()