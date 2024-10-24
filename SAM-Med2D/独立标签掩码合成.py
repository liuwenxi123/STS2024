import os
import cv2
import numpy as np

def map_value_to_grayscale(value, min_val=11, max_val=100, offset=50):
    # 对数字进行偏移
    shifted_value = value + offset
    # 确保值在合理范围内
    shifted_value = max(min_val + offset, min(max_val + offset, shifted_value))
    # 线性映射到0-255
    return int(255 * (shifted_value - (min_val + offset)) / (max_val - min_val))

def combine_masks(directory):
    # 获取目录下的所有PNG文件
    savename = os.path.basename(directory)
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    save_dir = r'D:\508\STS2024\SAM-Med2D-main\SAM_result\23f\pred'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 检查是否有文件
    if not files:
        print("没有找到任何PNG文件。")
        return

    # 读取第一个文件以获取图像尺寸
    first_file = os.path.join(directory, files[0])
    mask = cv2.imread(first_file, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历所有文件
    for filename in files:
        parts = filename.split('_')

        number_part = parts[6].split('.')[0]  # 提取文件名中的数字部分
        try:
            number = int(number_part)  # 将字符串转换成整数
            file_path = os.path.join(directory, filename)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 检查是否为二值图像
            if np.all((mask == 0) | (mask == 255)):
                # 将二值图像转换为具有特定灰度值的图像
                mapped_value = map_value_to_grayscale(number)
                combined_mask[mask == 255] = mapped_value
            else:
                print(f"警告：{filename} 不是二值图像。")
        except ValueError:
            print(f"警告：无法从 {filename} 中提取有效的数字。")

    # 显示和/或保存合成后的图像
    # cv2.imshow('Combined Masks', combined_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_path = os.path.join(save_dir, '{}.png'.format(savename))
    cv2.imwrite(output_path, combined_mask)
    print(f"合成后的掩码图像已保存到 {output_path}")


# 调用函数
if __name__ == '__main__':
    base_dir = r'D:\508\STS2024\SAM-Med2D-main\SAM_result\23f'
    for fd in os.listdir(base_dir):
        if fd.startswith('STS24'):
            folder = os.path.join(base_dir, fd)
            combine_masks(folder)