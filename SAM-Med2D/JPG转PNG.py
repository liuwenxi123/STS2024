from PIL import Image
import os

def convert_jpg_to_png(jpg_file_path, output_dir=None):
    """
    将给定路径的.jpg文件转换为.png文件。

    :param jpg_file_path: .jpg文件的完整路径。
    :param output_dir: 输出.png文件的目录，默认为None，表示输出到与.jpg相同的目录。
    :return: None
    """
    # 打开.jpg文件
    img = Image.open(jpg_file_path)

    # 如果没有指定输出目录，则保持原目录
    if output_dir is None:
        output_dir = os.path.dirname(jpg_file_path)

    # 构建输出文件名
    png_name = os.path.basename(jpg_file_path).replace('.jpg', '.png')
    png_path = os.path.join(output_dir, png_name)
    # 保存为.png格式
    img.save(png_path, 'PNG')
    print(f"Converted {jpg_file_path} to {png_path}")


if __name__ == '__main__':
    jpgs_dir = r'D:\508\STS2024\NewDataset\LabeledTrain\f\Images'
    pngsave_dir = r'D:\508\STS2024\NewDataset\LabeledTrain\f\ImagesPNG'
    if not os.path.exists(pngsave_dir):
        os.makedirs(pngsave_dir)

    for f in os.listdir(jpgs_dir):
        jpgfile_path = os.path.join(jpgs_dir, f)
        convert_jpg_to_png(jpgfile_path, pngsave_dir)