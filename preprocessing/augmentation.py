import json
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import shutil
import imageio
from scipy.ndimage import gaussian_filter
import h5py


def adjust_brightness_contrast(image_path, output_dir, brightness_factor=1, contrast_factor=1):
    """亮度对比度增强函数"""
    try:
        # 导入图像
        image = Image.open(image_path)
        # 调整亮度
        enhancer_brightness = ImageEnhance.Brightness(image)    # 亮度增强器
        image_brightened = enhancer_brightness.enhance(brightness_factor)
        # 调整对比度
        enhancer_contrast = ImageEnhance.Contrast(image_brightened)  # 对比度增强器
        image_final = enhancer_contrast.enhance(contrast_factor)
        # 输出路径
        output_filename = os.path.basename(image_path).replace('.jpg', '_aug.jpg')
        output_path = os.path.join(output_dir, output_filename)
        # 保存修改后的图像
        image_final.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def flip_label_from_json(json_path, save_path):
    """水平翻转json文件的坐标"""
    # 导入json文件
    with open(json_path, 'r') as f:
        label_data = json.load(f)
    # 翻转坐标
    height, width = label_data['imageHeight'], label_data['imageWidth']
    for shape in label_data['shapes']:
        list1 = [1, 3, 5, 7]
        list2 = [2, 4, 6, 8]
        label_index = int(shape['label'])
        ten_digit = (label_index // 10) % 10
        if ten_digit in list1:
            shape['label'] = str(int(shape['label']) + 10)
        elif ten_digit in list2:
            shape['label'] = str(int(shape['label']) - 10)

        for point in shape['points']:
            point[0] = width - point[0]

    # 保存文件

    with open(save_path, 'w') as file:
        json.dump(label_data, file, indent=4)  # 使用indent=4来美化输出格式

    # print(f"{os.path.basename(json_path)}已水平翻转为 {save_path}")


def flip_image_from_jpg(image_path, save_path):
    """水平翻转jpg文件"""
    with Image.open(image_path) as image:
        flipped_img = image.transpose(method=Image.FLIP_LEFT_RIGHT)

    flipped_img.save(save_path)


def lpfilter_image_from_jpg(image_path, output_dir):
    """低通滤波jpg图像"""
    lowpass_filter = ImageFilter.GaussianBlur(radius=5)
    with Image.open(image_path) as image:
        filtered_img = image.filter(lowpass_filter)

    output_filename = os.path.basename(image_path).replace('.jpg', '_flip.jpg')
    output_path = os.path.join(output_dir, output_filename)
    filtered_img.save(output_path)

    print(f"{os.path.basename(image_path)}已低通滤波保存为 {output_filename}")


def rotate_images_and_labels(images_dir, labels_dir, output_dir, rotation_angle=45):
    """旋转图像和标签的角度"""
    images_list = os.listdir(images_dir)
    labels_list = os.listdir(labels_dir)
    assert len(images_list) == len(labels_list)

    # output_dir = os.path.join(output_dir, 'rotated_{}'.format(rotation_angle))

    for i in range(len(images_list)):
        # 旋转图像
        with Image.open(os.path.join(images_dir, images_list[i])) as image:
            rotated_image = image.rotate(rotation_angle)
        # 旋转标签
        with open(os.path.join(labels_dir, labels_list[i]), 'r') as f:
            label_data = json.load(f)
            height, width = label_data['imageHeight'], label_data['imageWidth']
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        for shape in label_data['shapes']:
            for point in shape['points']:
                point[0], point[1] = np.dot(rotation_matrix, np.array([point[0], point[1], 1]))

        # 保存图像
        output_image_name = os.path.basename(images_list[i]).replace('.jpg', '_r{}.jpg'.format(rotation_angle))
        output_images_dir = os.path.join(output_dir, 'Images')
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
        output_path = os.path.join(output_images_dir, output_image_name)
        rotated_image.save(output_path)
        # 保存标签
        output_label_name = os.path.basename(labels_list[i]).replace('.json', '_r{}.json'.format(rotation_angle))
        output_labels_dir = os.path.join(output_dir, 'MasksJson')
        if not os.path.exists(output_labels_dir):
            os.makedirs(output_labels_dir)
        output_path = os.path.join(output_labels_dir, output_label_name)
        with open(output_path, 'w') as file:
            json.dump(label_data, file, indent=4)  # 使用indent=4来美化输出格式

    print('OJBK')


def rename_files(dir, ori_keyword, new_keyword, file_type):
    """批量对路径下文件进行关键字替换更名"""
    fileslist = os.listdir(dir)
    for filename in fileslist:
        if filename.lower().endswith(file_type):
            new_filename = filename.replace(ori_keyword, new_keyword)
            new_filepath = os.path.join(dir, new_filename)
            old_filepath = os.path.join(dir, filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")


def visualize_json(json_path, output_dir):
    """将json文件以图片的形式呈现出来"""
    assert os.path.exists(json_path), f"The path {json_path} does not exist."
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    height, width = label_data['imageHeight'], label_data['imageWidth']
    Masknp = np.zeros((53, height, width), np.uint8)

    for shape in label_data['shapes']:
        masknp = np.zeros([height, width], dtype=np.uint8)
        label = shape['label']
        label_index = int(label)
        ten_digit = (int(label) // 10) % 10  # 牙齿标签的十位数
        if ten_digit <= 4:
            label_index = label_index - (11 + (ten_digit - 1) * 2)
        else:
            label_index = label_index - (19 + (ten_digit - 5) * 5)

        points = np.array(shape['points'], dtype=np.int32)
        points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly

        cv2.fillPoly(masknp, [points], 53)
        Masknp[label_index] = masknp

    foreground_mask = np.max(Masknp[:51, :, :], axis=0)
    Masknp[52, :, :] = 1 - foreground_mask

    for i in range(Masknp.shape[0] - 1):
        Masknp[i, :, :] *= (i + 1)
    pred_background = Masknp[-1, :, :]
    pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
    Masknp[-1, :, :] = pred_background
    prediction = np.sum(Masknp, axis=0)
    prediction_uint8 = (prediction * 255).astype(np.uint8)

    json_name = os.path.basename(json_path)  # JSON文件名
    png_name = json_name.replace('.json', '.jpg')
    save_path = os.path.join(output_dir, png_name)
    imageio.imwrite(save_path, prediction_uint8)

    print('ojbk')


def equalize_histogram(image_path, save_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    eq_hist_img = cv2.equalizeHist(img)

    # Save the adjusted image
    cv2.imwrite(save_path, eq_hist_img)


def cl_ahe(image_path, save_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # You can adjust clipLimit and tileGridSize parameters
    cl_ahe_img = clahe.apply(img)

    # Save the adjusted image
    cv2.imwrite(save_path, cl_ahe_img)


def flip_h5(h5_path, save_path):
    """翻转h5"""
    with h5py.File(h5_path, 'r') as f:
        dataset = f['Mask']
        data_array = np.array(dataset)
        data = data_array.astype(int)
    flipped_data = np.flip(data, axis=2)
    mask_bool = flipped_data.astype(bool)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("Mask", data=mask_bool, dtype='bool', compression="gzip", compression_opts=9)



if __name__ == '__main__':
    # dir = r'D:\508\STS2024\SSL4MIS-master\data\unetres_epoch_137_dice_0.207\crop_transfer_h5_json1'
    # # for f in os.path.basename(dir):
    # #     file_path = os.path.join(dir, f)  STS24_Train_Validation_00001_pred
    # rename_files(dir, 'STS24_Train_Validation', 'Validation', 'json')
    # rename_files(dir, 'pred', 'Mask', 'json')
    # rename_files(dir, '_000', '_00', 'json')
    dir = r'D:\508\STS2024\dataset\Validation-Public-20240728T073621Z-001\crop'
    dir_save = dir+'_eq'
    os.makedirs(dir_save, exist_ok=True)
    for f in os.listdir(dir):
        fpath = os.path.join(dir, f)
        spath = os.path.join(dir_save, f)
        # adjust_brightness_contrast(fpath, dir_save, contrast_factor=1)
        equalize_histogram(fpath, spath)






    print('Done')

