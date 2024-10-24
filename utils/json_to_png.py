import os
# import bcolz
import cv2
import json
import math
import h5py
# import zarr
from h5_to_png import *

def json_draw(json_path, save_path, num_class=53):
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

    num_classes, height, width = Mask.shape[0], Mask.shape[1], Mask.shape[2]
    Masknp = np.zeros(Mask.shape, np.uint8)
    for label_index in range(1, num_classes):
        masknp = np.zeros((height, width), dtype=np.uint8)
        teeth_type = channels_dict[label_index]
        masknp[Mask[label_index]==1] = teeth_type  # 为每个标签设置不同的灰度值
        Masknp[label_index] = masknp
    foreground_mask = np.max(Masknp[1:num_classes, :, :], axis=0)
    Masknp[0, :, :] = 1 - foreground_mask
    pred_background = Masknp[0, :, :]
    pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
    Masknp[0, :, :] = pred_background
    prediction = np.sum(Masknp, axis=0)
    prediction_uint8 = (prediction * 255).astype(np.uint8)

    imageio.imwrite(save_path, prediction_uint8)

import numpy as np
def np_draw(nparray, save_path):
    data_array = nparray
    num_classes, height, width = data_array.shape[0], data_array.shape[1], data_array.shape[2]
    Masknp = np.zeros(data_array.shape, dtype=np.uint8)
    for label_index in range(1, num_classes):
        masknp = np.zeros((height, width), dtype=np.uint8)
        teeth_type = channels_dict[label_index]
        masknp[data_array[label_index]==1] = teeth_type  # 为每个标签设置不同的灰度值
        Masknp[label_index] = masknp
    foreground_mask = np.max(Masknp[1:num_classes, :, :], axis=0)
    Masknp[0, :, :] = 1 - foreground_mask
    pred_background = Masknp[0, :, :]
    pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
    Masknp[0, :, :] = pred_background
    prediction = np.sum(Masknp, axis=0)
    prediction_uint8 = (prediction * 255).astype(np.uint8)

    imageio.imwrite(save_path, prediction_uint8)

