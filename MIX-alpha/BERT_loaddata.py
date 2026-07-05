import json
import os
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset
from transformers import BertTokenizer

# 初始化 tokenizer（只初始化一次）
tokenizer = BertTokenizer.from_pretrained('D:/bert-base-chinese')


class TextAndLabelDataset(Dataset):
    def __init__(self, json_dir, pt_label_dir, file_list=None):
        """
        json_dir: 存放 JSON 文本数据的目录
        pt_label_dir: 存放 PT 标签数据的目录
        file_list: 指定使用的文件列表，用于划分数据集
        """
        self.json_dir = json_dir
        self.pt_label_dir = pt_label_dir

        # 获取指定的JSON文件列表或所有JSON文件
        if file_list is not None:
            self.json_files = file_list
        else:
            self.json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

        # 检查对应的 PT 文件是否存在
        for json_file in self.json_files:
            basename = json_file.replace('.json', '')
            pt_file = basename + '.pt'
            pt_path = os.path.join(pt_label_dir, pt_file)
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"找不到标签文件：{pt_path}，请确保 processed/ 目录下有与 JSON 同名的 .pt 文件")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # 1. 加载 JSON 文本数据
        json_file = self.json_files[idx]
        json_path = os.path.join(self.json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        text = json_data.get('text', '')  # 假设 JSON 里有 "text" 字段

        # 2. 加载 PT 标签数据（图数据，里面含有 data.y 标签）
        basename = json_file.replace('.json', '')
        pt_file = basename + '.pt'
        pt_path = os.path.join(self.pt_label_dir, pt_file)
        data = torch.load(pt_path, weights_only=False)  # 这是一个 PyG 的 Data 对象
        label = data.y  # 标签，比如 tensor([1]) 或 tensor(0)

        # 3. 对文本进行 BERT Tokenizer 编码
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,  # 你可以调整这个长度
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [seq_len]
        token_type_ids = encoded['token_type_ids'].squeeze(0)  # [seq_len]，对于单句一般为全0

        label = label.squeeze()  # 转为标量 tensor，如 tensor(1)

        return input_ids, attention_mask, token_type_ids, label


import torch
import torch.nn as nn
from transformers import BertModel


class model(nn.Module):
    # ★ 修改点：增加 bert_path 参数，并放在第一位
    def __init__(self, bert_path, num_classes=2):
        super(model, self).__init__()

        # 第一层：BERT 模型
        # ★ 修改点：使用传入的 bert_path，而不是写死的字符串
        self.bert = BertModel.from_pretrained(bert_path)

        # 冻结前8层 (保持你原有的逻辑)
        '''for layer in self.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False'''

        # 第二层：全连接层
        # BERT-base-chinese 的隐藏层维度是 768
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 如果你没有传入 token_type_ids，默认 BERT 也可以处理
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 取 [CLS] token 的输出
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 通过分类层
        logits = self.classifier(cls_output)

        return logits