import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import json
import math
import h5py
from utils.teeth_idx_dict import *

class Dataset(Dataset):
    def __init__(self, images, labels):
        super(Dataset).__init__()
        self.images = images  # 图像数据集
        self.labels = labels  # 标签数据集
        self.num_class = 53   # 分类数

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 读取图像和标签
        imagesample = self.images[index]
        labelsample = self.labels[index]

        # 预处理图像
        ori_image = cv2.imread(imagesample, 0)  # 读取原始图像(h,w)
        ori_image = cropping(ori_image)
        image = cv2.resize(ori_image, (1024,1024))  # (h,w)->(1024,1024)
        image = (image - image.mean()) / image.std()    # 归一化
        image = image.reshape(1, 1024, 1024)    # (1024,1024)->(1,1024,1024)，二维变三维，添加通道维度1
        images_tensor = torch.as_tensor(image).float()  # 转成张量

        # 预处理标签(json)
        mask = self.load_label_from_json(labelsample)  # 已Resize
        if mask is not None:
            label_tensor = torch.as_tensor(mask).float()
        else:
            label_tensor = None

        sample = {"image": images_tensor, "label": label_tensor, "origin_size": ori_image.shape}
        return sample


    def load_label_from_h5(self, h5_path):
        """从h5文件中获取数据，并Resize为指定尺寸"""
        if isinstance(h5_path, float):
            if math.isnan(h5_path):
                return np.zeros((53, 1024, 1024))

        with h5py.File(h5_path, 'r') as f:
            dataset = f['Mask']
            data_array = np.array(dataset, dtype=int)

        num_classes = data_array.shape[0]
        resized_data_array = np.empty((data_array.shape[0], 1024, 1024), dtype=int)

        for i in range(num_classes):
            mask_temp = data_array[i]
            resized_mask = cv2.resize(mask_temp, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            resized_data_array[i] = resized_mask

        return resized_data_array


    def load_label_from_json(self, json_path):
        """读取JSON文件内的坐标，并处理为np数组，大小为(53,1024,1024)"""
        if isinstance(json_path, float):
            if math.isnan(json_path):
                return np.zeros((53, 1024, 1024))
        if not json_path:
            return np.zeros((53, 1024, 1024))

        with open(json_path, 'r') as f:
            label_data = json.load(f)

        height, width = label_data['imageHeight'], label_data['imageWidth']
        Mask = np.zeros((53, 1024, 1024))

        for shape in label_data['shapes']:
            mask_temp = np.zeros((height, width), dtype=np.uint8)
            label = shape['label']
            label_index = teeth_dict[int(label)]
            points = np.array(shape['points'], dtype=np.int32)
            points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly
            cv2.fillPoly(mask_temp, [points], 1)
            mask_temp = cv2.resize(mask_temp, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            Mask[label_index] = mask_temp

        foreground_mask = np.max(Mask[:-1, :, :], axis=0)
        Mask[-1, :, :] = 1 - foreground_mask

        return Mask

def cl_ahe(image_np, clipLimit=2.0, tileGridSize=(8, 8)):
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    # 应用CLAHE
    cl_ahe_img_np = clahe.apply(image_np)
    return cl_ahe_img_np

def cropping(image_np):
    height, width = image_np.shape
    x_start, y_start = 100, 50
    x_end, y_end = width - 100, height - 50
    cropped_image = image_np[y_start:y_end, x_start:x_end]
    return cropped_image

