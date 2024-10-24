import json
import os
from PIL import Image, ImageDraw
import numpy as np

jsonmasks_dir = r'D:\508\STS2024\NewDataset\LabeledTrain\f\MasksJson'
savemaskspng_dir = r'D:\508\STS2024\NewDataset\LabeledTrain\f\MasksPNG'
if not os.path.exists(savemaskspng_dir):
    os.makedirs(savemaskspng_dir)

def create_mask(json_path):
    # 从json_data中提取图像的高度和宽度
    json_data = json.load(open(json_path, 'r'))

    image_height = json_data['imageHeight']
    image_width = json_data['imageWidth']

    # 遍历每个shape
    for shape in json_data['shapes']:
        label = shape['label']
        points = shape['points']

        mask_image = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask_image)

        flat_points = flatten_points(points)
        # 转换points为可绘制的格式
        draw.polygon(flat_points, outline=1, fill=1)

        # 保存每个label对应的掩膜图像
        mask_array = np.array(mask_image)
        mask_image_for_label = Image.fromarray((mask_array == 1).astype(np.uint8) * 255)

        png_name = os.path.basename(json_path).replace('.json', f'_{label}.png')
        mask_image_for_label.save(os.path.join(savemaskspng_dir, png_name))


def flatten_points(points):
    """Flatten a list of points from [[x1, y1], [x2, y2], ...] to [x1, y1, x2, y2, ...]."""
    return [point for sublist in points for point in sublist]


if __name__ == '__main__':
    for f in os.listdir(jsonmasks_dir):
        json_file = os.path.join(jsonmasks_dir, f)
        create_mask(json_file)
