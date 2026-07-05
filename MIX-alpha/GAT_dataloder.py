import os
import torch
from torch_geometric.data import InMemoryDataset


class InMemoryWeiboDataset(InMemoryDataset):
    """
    兼容你新的图数据集（每个微博一个 .pt 图文件）
    root/
        processed/
            xxx.pt
            yyy.pt
    """

    def __init__(self, root, bert_folder=None, transform=None, pre_transform=None):
        self.bert_folder = bert_folder
        super().__init__(root, transform, pre_transform)

        # 1. 获取排序后的文件名列表 (保证顺序固定)
        file_names = self.processed_file_names

        # 2. ★关键修改：显式保存文件名列表，供外部索引使用
        self.filenames = file_names

        # 3. 加载数据
        data_list = []
        for name in file_names:
            path = os.path.join(self.processed_dir, name)
            data = torch.load(path, weights_only=False)

            # 确保 sample_id 存在
            if not hasattr(data, "sample_id"):
                data.sample_id = os.path.splitext(name)[0]

            data_list.append(data)

        # InMemoryDataset 规范：通过 collate 将 data_list 打包
        self.data, self.slices = self.collate(data_list)

    # ---------------------------------------------------------------------
    # 处理后的文件（processed 目录下所有 .pt 文件）
    # ---------------------------------------------------------------------
    @property
    def processed_file_names(self):
        """processed/ 下所有图文件"""
        names = []
        if os.path.exists(self.processed_dir):
            for fn in os.listdir(self.processed_dir):
                if fn.endswith(".pt"):
                    names.append(fn)
        # ★必须排序，保证每次加载顺序一致
        return sorted(names)

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass