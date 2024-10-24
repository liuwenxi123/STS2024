import os
from PIL import Image
import numpy as np
import cv2
import h5py
import random


def rotate_images_and_labels(images_dir, labels_dir, output_dir):
    """旋转图像和标签的角度，旋转角度为随机正负10度"""
    images_list = os.listdir(images_dir)
    labels_list = os.listdir(labels_dir)
    assert len(images_list) == len(labels_list)

    for i in range(len(images_list)):
        # 生成随机旋转角度
        rotation_angle = random.uniform(-10, 10)

        # 旋转图像
        with Image.open(os.path.join(images_dir, images_list[i])) as image:
            rotated_image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

        # 保存图像
        output_image_name = os.path.basename(images_list[i]).replace('.jpg', f'_r{rotation_angle:.1f}.jpg')
        output_images_dir = os.path.join(output_dir, 'Images')
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
        output_path = os.path.join(output_images_dir, output_image_name)
        rotated_image.save(output_path)

        # 旋转标签
        h5path = os.path.join(labels_dir, labels_list[i])
        with h5py.File(h5path, 'r') as f:
            # 从文件中读取名为 "Mask" 的数据集
            dataset = f['Mask']
            data_array = np.array(dataset)
            h5_array = data_array.astype(int)
        h, w = data_array.shape[1], data_array.shape[2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_data = np.zeros_like(h5_array)
        for j in range(h5_array.shape[0]):
            # 旋转当前平面
            rotated_plane = cv2.warpAffine(h5_array[j], rotation_matrix, (w, h), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # 将旋转后的平面放入结果数组中
            rotated_data[j] = rotated_plane

        # 保存标签
        mask_bool = rotated_data.astype(bool)
        h5py_save_dir = os.path.join(output_dir, 'MasksH5')
        if not os.path.exists(h5py_save_dir):
            os.makedirs(h5py_save_dir)
        h5py_save_path = os.path.join(h5py_save_dir, labels_list[i].replace('.h5', f'_r{rotation_angle:.1f}.h5'))
        with h5py.File(h5py_save_path, 'w') as f:
            f.create_dataset("Mask", data=mask_bool, dtype='bool', compression="gzip", compression_opts=9)


# 使用示例
images_dir = r'D:\508\STS2024\TeethDataset\Train-Labeled\Images'
labels_dir = r'D:\508\STS2024\TeethDataset\Train-Labeled\MasksH5'
output_dir = r'D:\508\STS2024\TeethDataset\rotated'

rotate_images_and_labels(images_dir, labels_dir, output_dir)