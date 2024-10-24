import torch
import torch.nn as nn
from model.Teeth53_Model import Teeth53_Model

# 创建模型
model = Teeth53_Model(in_channels=1, out_classes=53)

# 输入数据
input_tensor = torch.randn(4, 1, 1024, 1024)  # batch_size=4, channels=1, height=1024, width=1024

# 计算参数数量
num_params = sum(p.numel() for p in model.parameters())
print(f'Number of model parameters: {num_params}')

# 计算FLOPs
def count_flops(model, input_tensor):
    total_flops = 0  # Define total_flops here

    def flops_hook(module, input, output):
        nonlocal total_flops  # Use nonlocal to modify total_flops
        batch_size = input[0].size(0)

        if isinstance(module, nn.Conv2d):
            # 获取卷积层的参数
            out_channels, in_channels, kernel_h, kernel_w = module.weight.size()
            stride_h, stride_w = module.stride[0], module.stride[1]  # 获取步长
            padding_h, padding_w = module.padding[0], module.padding[1]  # 获取填充
            dilation_h, dilation_w = module.dilation[0], module.dilation[1]  # 获取膨胀

            # 计算输出特征图的尺寸
            output_h, output_w = output.size()[2], output.size()[3]

            # 计算卷积操作的次数
            flops_per_instance = (output_h * output_w) * kernel_h * kernel_w * in_channels * out_channels
            # 乘以批量大小
            flops = batch_size * flops_per_instance
            # 累加FLOPs
            total_flops += flops

    for layer in model.modules():
        layer.register_forward_hook(flops_hook)

    with torch.no_grad():
        model(input_tensor)

    return total_flops

num_flops = count_flops(model, input_tensor)
print(f'Number of FLOPs: {num_flops / 1e9:.2f} G')

# 计算COzeq
cozeq = num_flops / num_params
print(f'COzeq: {cozeq:.2f}')