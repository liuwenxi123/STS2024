# -*- coding: utf-8 -*-
"""
    Author：Teddy
    filename： ConnectedRegion_np.py
    Created on：2024/9/22 17:21
"""
import numpy as np
import scipy.ndimage as ndi
import h5py
import os

def retain_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)

    # 查找所有连通域
    labeled_mask, num_labels = ndi.label(mask)

    if num_labels == 0:
        # 如果没有连通域，返回全零的掩码
        return np.zeros_like(mask, dtype=np.uint8)

    # 计算每个连通域的面积
    region_areas = np.bincount(labeled_mask.ravel())[1:]  # 忽略背景（0值）
    largest_region_label = region_areas.argmax() + 1  # 获取最大连通域的标签

    if region_areas[largest_region_label - 1] <= 500:
        return np.zeros_like(mask, dtype=np.uint8)

    # 创建一个新的掩码，只保留最大连通域
    largest_region_mask = (labeled_mask == largest_region_label).astype(np.uint8)

    # 获取图像的宽度
    image_width = mask.shape[1]
    left_threshold = image_width // 6
    right_threshold = 5 * image_width // 6
    up_threshold = 200
    down_threshold = mask.shape[0] - 200

    # 计算最大连通域的边界框
    y, x = np.where(largest_region_mask)
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    # 检查最大连通域是否主要位于左或右1/6区域
    # if max_x < left_threshold or min_x > right_threshold:
    #     return np.zeros_like(mask, dtype=np.uint8)
    if min_y < up_threshold or max_y > down_threshold:
        return np.zeros_like(mask, dtype=np.uint8)

    return largest_region_mask

import numpy as np
from scipy import ndimage as ndi


def filter_regions(mask: np.ndarray, min_area: int = 1000, up_threshold: int = 200, down_threshold: int = None, left_threshold: int = 150, right_threshold: int = None) -> np.ndarray:
    # 查找所有连通域
    labeled_mask, num_labels = ndi.label(mask)

    if num_labels == 0:
        # 如果没有连通域，返回全零的掩码
        return np.zeros_like(mask)

    # 计算每个连通域的面积
    region_areas = np.bincount(labeled_mask.ravel())[1:]  # 忽略背景（0值）

    # 获取图像的高度和宽度
    image_height, image_width = mask.shape
    if down_threshold is None:
        down_threshold = image_height - up_threshold
    if right_threshold is None:
        right_threshold = image_width - left_threshold

    # 创建一个新的掩码，用于存储符合条件的连通域
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels + 1):
        if region_areas[label - 1] <= min_area:
            continue  # 剔除面积太小的区域

        # 获取当前连通域的边界框
        y, x = np.where(labeled_mask == label)
        min_y, max_y = y.min(), y.max()
        min_x, max_x = x.min(), x.max()

        # 检查当前连通域是否位于上下阈值之外
        if min_y < up_threshold or max_y > down_threshold:
            continue  # 剔除太上或太下的区域

        # 检查当前连通域是否位于左右阈值之外
        if min_x < left_threshold or max_x > right_threshold:
            continue  # 剔除太左或太右的区域

        # 保留当前连通域
        filtered_mask[labeled_mask == label] = 1

    return filtered_mask


if __name__ == '__main__':
    from teeth_idx_dict import *
    import imageio
    dir = r'D:\508\STS2024\STS53_lwx\outputs_H5'
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        with h5py.File(file_path, 'r') as f:
            # 从文件中读取名为 "Mask" 的数据集
            dataset = f['Mask']
            data_array = np.array(dataset)
            data_array = data_array.astype(int)

        new_hard_mask = np.zeros((data_array.shape[0], data_array.shape[1], data_array.shape[2]), np.uint8)
        for j in range(data_array.shape[0]):
            new_hard_mask[j] = filter_regions(data_array[j])
        data_array = new_hard_mask


        num_classes, height, width = data_array.shape[0], data_array.shape[1], data_array.shape[2]
        Masknp = np.zeros(data_array.shape, np.uint8)
        for label_index in range(1, num_classes):
            masknp = np.zeros((height, width), dtype=np.uint8)
            teeth_type = channels_dict[label_index]
            masknp[data_array[label_index] == 1] = teeth_type  # 为每个标签设置不同的灰度值
            Masknp[label_index] = masknp
        foreground_mask = np.max(Masknp[1:num_classes, :, :], axis=0)
        Masknp[0, :, :] = 1 - foreground_mask
        pred_background = Masknp[0, :, :]
        pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
        Masknp[0, :, :] = pred_background
        prediction = np.sum(Masknp, axis=0)
        prediction_uint8 = (prediction * 255).astype(np.uint8)

        os.makedirs(dir+'_png', exist_ok=True)
        output_path = os.path.join(dir+'_png', file.replace('.h5', '.png'))
        imageio.imwrite(output_path, prediction_uint8)



