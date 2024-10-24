# -*- coding: utf-8 -*-
"""
    Author：Teddy
    filename： crop.py
    Created on：2024/9/20 16:58
"""
import os
import h5py
from PIL import Image
import numpy as np

# 定义数据路径
images_path = r'D:\508\STS2024\TeethDataset\Train-Labeled\Images'
masks_path = r'D:\508\STS2024\TeethDataset\Train-Labeled\MasksH5'
crop_images_path = r'D:\508\STS2024\TeethDataset\crop\Images'
crop_masks_path = r'D:\508\STS2024\TeethDataset\crop\MasksH5'

# 创建保存crop图像的目录
os.makedirs(crop_images_path, exist_ok=True)
os.makedirs(crop_masks_path, exist_ok=True)


# 裁剪函数，给定图像数组和自定义区域的坐标
def crop_image(image_array):
    height, width = image_array.shape[:2]

    # 自定义裁剪区域的比例
    crops = {
        "top_left": image_array[:int(height * 0.9), :int(width * 0.9)],
        "top_right": image_array[:int(height * 0.9), int(width * 0.1):],
        "bottom_left": image_array[int(height * 0.1):, :int(width * 0.9)],
        "bottom_right": image_array[int(height * 0.1):, int(width * 0.1):]
    }
    return crops

def crop_mask(image_array):
    height, width = image_array.shape[1], image_array.shape[2]

    # 自定义裁剪区域的比例
    crops = {
        "top_left": image_array[:, :int(height * 0.9), :int(width * 0.9)],
        "top_right": image_array[:, :int(height * 0.9), int(width * 0.1):],
        "bottom_left": image_array[:, int(height * 0.1):, :int(width * 0.9)],
        "bottom_right": image_array[:, int(height * 0.1):, int(width * 0.1):]
    }
    return crops

# 处理图像的函数
def process_images_and_masks():
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    for image_file, mask_file in zip(image_files, mask_files):
        # 加载图像
        image = Image.open(os.path.join(images_path, image_file))
        image_array = np.array(image)

        # 加载掩码
        with h5py.File(os.path.join(masks_path, mask_file), 'r') as f:
            # 从文件中读取名为 "Mask" 的数据集
            dataset = f['Mask']
            data_array = np.array(dataset)
            data_array = data_array.astype(int)

        # 对图像和掩码进行裁剪
        image_crops = crop_image(image_array)
        mask_crops = crop_mask(data_array)

        # 保存裁剪结果
        for key in image_crops:
            # 保存图像
            cropped_image = Image.fromarray(image_crops[key])
            cropped_image.save(os.path.join(crop_images_path, f"{os.path.splitext(image_file)[0]}_{key}.png"))

            # 保存掩码
            cropped_mask = mask_crops[key]
            mask_bool = cropped_mask.astype(bool)
            h9py_path = os.path.join(crop_masks_path, f"{os.path.splitext(mask_file)[0]}_{key}.h5")
            with h5py.File(h9py_path, 'w') as f:
                f.create_dataset("Mask", data=mask_bool, dtype='bool', compression="gzip", compression_opts=9)


process_images_and_masks()
