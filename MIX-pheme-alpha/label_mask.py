import pandas as pd
import numpy as np
import random
import argparse
import os


def split_and_mask_labels(
        input_csv,
        train_output_csv,
        val_output_csv, # 新增：验证集输出路径
        test_output_csv,
        seed=42,
        mask_ratio=0.0,
        per_class=False,
        test_size=0.2,
        val_size=0.1 # 新增：验证集比例
):
    """
    划分训练集、验证集和测试集。
    对训练集标签进行屏蔽后输出，验证集和测试集保持原始标签。

    input_csv: 原始标签文件
    train_output_csv: 训练集（含屏蔽标签）输出路径
    val_output_csv: 验证集（原始标签）输出路径
    test_output_csv: 测试集（原始标签）输出路径
    mask_ratio: 训练集标签屏蔽比例（可调整）
    per_class: 是否在类别内随机屏蔽
    test_size: 测试集比例
    val_size: 验证集比例
    """
    if val_size + test_size >= 1.0:
        raise ValueError("验证集比例 (val_size) 和测试集比例 (test_size) 之和必须小于 1.0")

    random.seed(seed)
    np.random.seed(seed)

    # 读取原始数据
    df = pd.read_csv(input_csv, dtype=str)
    id_col = df.columns[0]
    label_col = df.columns[1]
    df[label_col] = df[label_col].astype(int)

    # 划分训练集、验证集和测试集
    if per_class:
        # 按类别分层抽样，保持类别分布
        train_indices = []
        val_indices = []
        test_indices = []
        unique_labels = df[label_col].unique().tolist()

        for lab in unique_labels:
            if lab == -100:
                continue
            cls_idx = df[df[label_col] == lab].index.tolist()
            random.shuffle(cls_idx)

            # 计算各集合大小
            test_n = int(len(cls_idx) * test_size)
            val_n = int(len(cls_idx) * val_size)

            # 划分
            test_indices.extend(cls_idx[:test_n])
            val_indices.extend(cls_idx[test_n:test_n + val_n])
            train_indices.extend(cls_idx[test_n + val_n:])
    else:
        # 随机抽样划分
        all_indices = list(df.index)
        random.shuffle(all_indices)

        # 计算各集合大小
        test_n = int(len(all_indices) * test_size)
        val_n = int(len(all_indices) * val_size)

        # 划分
        test_indices = all_indices[:test_n]
        val_indices = all_indices[test_n:test_n + val_n]
        train_indices = all_indices[test_n + val_n:]

    # 分离训练集、验证集和测试集
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy() # 验证集保持原始数据
    test_df = df.loc[test_indices].copy() # 测试集保持原始数据

    # 仅对训练集标签进行屏蔽
    if per_class:
        # 按类别内屏蔽
        unique_labels = train_df[label_col].unique().tolist()
        for lab in unique_labels:
            if lab == -100:
                continue
            cls_idx = train_df[train_df[label_col] == lab].index.tolist()
            random.shuffle(cls_idx)
            mask_n = int(len(cls_idx) * mask_ratio)
            mask_idx = cls_idx[:mask_n]
            train_df.loc[mask_idx, label_col] = -100
    else:
        # 直接随机屏蔽
        train_idx = list(train_df.index)
        random.shuffle(train_idx)
        mask_n = int(len(train_idx) * mask_ratio)
        mask_idx = train_idx[:mask_n]
        train_df.loc[mask_idx, label_col] = -100

    # 保存结果
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False) # 保存验证集
    test_df.to_csv(test_output_csv, index=False)

    # 打印统计信息
    print(f"总样本数: {len(df)}")
    print("--- 划分和屏蔽结果 ---")
    print(f"训练集样本数: {len(train_df)} (屏蔽比例 {mask_ratio * 100}%)")
    print(f"训练集实际屏蔽样本数: {(train_df[label_col] == -100).sum()}")
    print(f"验证集样本数: {len(val_df)} (全部为原始标签)")
    print(f"测试集样本数: {len(test_df)} (全部为原始标签)")
    print(f"训练集（含屏蔽标签）已保存至: {train_output_csv}")
    print(f"验证集（原始标签）已保存至: {val_output_csv}")
    print(f"测试集（原始标签）已保存至: {test_output_csv}")


if __name__ == "__main__":
    # 参数设置
    input_path = r"D:\project\data\PHEME_Dataset\full_label.csv"  # 原始标签路径
    train_output_path = r"D:\project\data\PHEME_Dataset\train_label.csv"  # 训练集输出路径
    val_output_path = r"D:\project\data\PHEME_Dataset\val_label.csv"  # 新增：验证集输出路径
    test_output_path = r"D:\project\data\PHEME_Dataset\test_label.csv"  # 测试集输出路径
    seed = 42  # 随机种子
    mask_ratio = 0.9665  # 训练集标签屏蔽比例
    per_class = False  # 是否按类别处理（划分和屏蔽）
    test_size = 0.2  # 测试集比例（20%）
    val_size = 0.1  # 新增：验证集比例（10%）

    # 执行函数
    split_and_mask_labels(
        input_csv=input_path,
        train_output_csv=train_output_path,
        val_output_csv=val_output_path,
        test_output_csv=test_output_path,
        seed=seed,
        mask_ratio=mask_ratio,
        per_class=per_class,
        test_size=test_size,
        val_size=val_size
    )