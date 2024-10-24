# -*- coding: utf-8 -*-
"""
    Author：Teddy
    filename： ConnectedRegion_h5.py
    Created on：2024/9/22 17:17
"""
import h5py
import numpy as np
import os
import scipy.ndimage as ndi

# 文件路径
input_folder = 'MaskH5'
output_folder = 'Save'

# 创建保存结果的文件夹，如果不存在
os.makedirs(output_folder, exist_ok=True)


def process_h5_file(file_path, output_path):
    with h5py.File(file_path, 'r') as f:
        # 读取 "Mask" 数据集
        mask = np.array(f['Mask'], dtype=np.uint8)

    # 假设 mask 为二值图，0 为背景，1 为前景
    # 查找所有连通域
    labeled_mask, num_labels = ndi.label(mask)

    if num_labels == 0:
        # 如果没有连通域，创建全零掩码
        largest_region_mask = np.zeros_like(mask, dtype=np.uint8)
        print(f"No regions found in {file_path}, saving empty mask.")
    else:
        # 计算每个连通域的面积
        region_areas = np.bincount(labeled_mask.ravel())[1:]  # 忽略背景（0值）
        largest_region_label = region_areas.argmax() + 1  # 获取最大连通域的标签

        # 创建一个新的掩码，只保留最大连通域
        largest_region_mask = (labeled_mask == largest_region_label).astype(np.uint8)
        print(f"Processed {file_path}, largest region found and saved.")

    # 将处理结果保存为 .h5 文件
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('Mask', data=largest_region_mask, dtype=np.uint8)


def process_all_h5_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.h5'):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_h5_file(file_path, output_path)


if __name__ == '__main__':
    process_all_h5_files(input_folder, output_folder)
