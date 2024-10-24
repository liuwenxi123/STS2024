import os
# import bcolz
import cv2
import numpy as np
import json
import math
import h5py
# import zarr
from h5_to_png import *

def from_json_to_h5py(json_path, h9py_path, num_class):
    """读取JSON文件内的坐标，并处理为np数组，大小为(53,1024,1024)"""
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    height, width = label_data['imageHeight'], label_data['imageWidth']

    Mask = np.zeros((num_class, height, width))

    for shape in label_data['shapes']:
        mask_temp = np.zeros((height, width), dtype=np.uint8)
        label = shape['label']
        label_index = int(label)
        ten_digit = label_index // 10  # 牙齿标签的十位数
        if ten_digit <= 4:
            new_label_index = label_index - (10 + (ten_digit - 1) * 2)
        else:
            new_label_index = label_index - (18 + (ten_digit - 5) * 5)

        points = np.array(shape['points'], dtype=np.int32)
        points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly
        cv2.fillPoly(mask_temp, [points], 1)
        # mask_temp = cv2.resize(mask_temp, (height, width), interpolation=cv2.INTER_NEAREST)
        Mask[new_label_index] = mask_temp

    foreground_mask = np.max(Mask[1:num_class, :, :], axis=0)
    Mask[0, :, :] = 1 - foreground_mask

    mask_bool = Mask.astype(bool)
    with h5py.File(h9py_path, 'w') as f:
        f.create_dataset("Mask", data=mask_bool, dtype='bool', compression="gzip", compression_opts=9)


if __name__ == '__main__':
    json_dir = r'D:\508\STS2024\STS53_lwx\outputs'
    h5py_dir = json_dir + '_H5'
    if not os.path.exists(h5py_dir):
        os.mkdir(h5py_dir)
    for f in os.listdir(json_dir):
        if f.endswith('.json'):
            json_path = os.path.join(json_dir, f)
            h5py_path = os.path.join(h5py_dir, f.replace('.json', '.h5'))
            from_json_to_h5py(json_path, h5py_path, 53)

    npy_dir = h5py_dir
    save_dir = npy_dir + '_png'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for f in os.listdir(npy_dir):
        file_path = os.path.join(npy_dir, f)
        save_path = os.path.join(save_dir, f.replace('.h5', '.png'))
        loaded_array = load_h5_file(file_path, save_path)