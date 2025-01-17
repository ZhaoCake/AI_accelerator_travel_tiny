import torch
import numpy as np
from net_code import SimpleCNN
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_model(weights_path='mnist_cnn.pth'):
    """加载训练好的模型"""
    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

def preprocess_image(image_path):
    """预处理输入图像"""
    # 读取图像
    image = Image.open(image_path)
    
    # 检查并打印原始图像信息
    print(f"Original image size: {image.size}")
    print(f"Original image mode: {image.mode}")
    
    # 处理不同的图像模式
    if image.mode == '1':  # 二值图像
        image = image.convert('L')
        print("Converted binary to grayscale")
    elif image.mode == 'RGB':
        image = image.convert('L')
        print("Converted RGB to grayscale")
    elif image.mode != 'L':
        raise ValueError(f"Unexpected image mode: {image.mode}")
    
    # 转换为numpy数组并检查值范围
    image_np = np.array(image, dtype=np.float32)
    print(f"Value range before normalization: [{image_np.min()}, {image_np.max()}]")
    
    # 如果值范围不是0-255，进行调整
    if image_np.max() == 1:
        image_np *= 255
        print("Adjusted value range to 0-255")
    
    # 转回PIL图像
    image = Image.fromarray(image_np.astype(np.uint8))
    
    # MNIST标准化处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 除以255并转换为[0,1]
        transforms.Resize((32, 32), antialias=True),  # 调整大小为32x32
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])
    
    # 转换为tensor
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 打印处理后的信息
    print(f"Processed tensor shape: {image_tensor.shape}")
    print(f"Value range after normalization: [{image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}]")
    
    return image_tensor

def save_array_to_dat(arr, filename):
    """将数组保存为逗号分隔的文本格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 将数组展平并转换为字符串
        data_str = ','.join(f"{x:.10f}" for x in arr.flatten())
        # 添加花括号包裹
        f.write(data_str)

def quantize_for_hls(data, bits=16, int_bits=8):
    """量化数据为定点数格式"""
    scale = 2.0 ** (bits - int_bits - 1)
    return np.clip(np.round(data * scale), -2**(bits-1), 2**(bits-1)-1) / scale

def inference_and_save(model, image_path, save_dir='test_data'):
    """进行推理并保存量化后的输入数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 预处理图像并打印信息
    print("\nPreprocessing image...")
    image = preprocess_image(image_path)
    
    # 获取numpy数组并量化
    image_np = image.numpy()
    print(f"Input shape before transpose: {image_np.shape}")
    
    # 转换为HWC格式 (1,1,32,32) -> (32,32,1)
    image_np = np.transpose(image_np[0], (1,2,0))
    print(f"Input shape after transpose: {image_np.shape}")
    
    # 量化数据
    quantized_data = quantize_for_hls(image_np)
    print(f"Quantized data range: [{quantized_data.min()}, {quantized_data.max()}]")
    
    # 保存量化后的输入数据
    save_array_to_dat(quantized_data, f'{save_dir}/input.dat')
    
    # 保存一个可视化的版本用于检查
    plt.imsave(f'{save_dir}/input_visualization.png', 
              quantized_data.squeeze(), 
              cmap='gray')
    
    # 进行推理并保存中间结果
    with torch.no_grad():
        x = image
        
        # conv1 + relu (NCHW -> HWC)
        x = F.relu(model.conv1(x))
        x_np = quantize_for_hls(x.numpy())
        x_np = np.transpose(x_np[0], (1,2,0))  # (1,16,32,32) -> (32,32,16)
        save_array_to_dat(x_np, f'{save_dir}/conv1_output.dat')
        
        # pool1
        x = model.pool1(x)
        x_np = quantize_for_hls(x.numpy())
        x_np = np.transpose(x_np[0], (1,2,0))  # (1,16,16,16) -> (16,16,16)
        save_array_to_dat(x_np, f'{save_dir}/pool1_output.dat')
        
        # conv2 + relu
        x = F.relu(model.conv2(x))
        x_np = quantize_for_hls(x.numpy())
        x_np = np.transpose(x_np[0], (1,2,0))  # (1,32,16,16) -> (16,16,32)
        save_array_to_dat(x_np, f'{save_dir}/conv2_output.dat')
        
        # pool2
        x = model.pool2(x)
        x_np = quantize_for_hls(x.numpy())
        x_np = np.transpose(x_np[0], (1,2,0))  # (1,32,8,8) -> (8,8,32)
        save_array_to_dat(x_np, f'{save_dir}/pool2_output.dat')
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc1 + relu
        x = F.relu(model.fc1(x))
        x_np = quantize_for_hls(x.numpy())
        save_array_to_dat(x_np, f'{save_dir}/fc1_output.dat')
        
        # fc2
        x = model.fc2(x)
        output = quantize_for_hls(x.numpy())
        save_array_to_dat(output, f'{save_dir}/fc2_output.dat')
        
        # 计算最终结果
        prob = torch.nn.functional.softmax(torch.from_numpy(output), dim=1)
        pred = output.argmax(axis=1)
    
    # 保存形状信息
    shapes = {
        'input': (32,32,1),    # HWC
        'conv1': (32,32,16),   # HWC
        'pool1': (16,16,16),   # HWC
        'conv2': (16,16,32),   # HWC
        'pool2': (8,8,32),     # HWC
        'fc1': (1,128),        # NC
        'fc2': (1,10)          # NC
    }
    with open(f'{save_dir}/shapes.txt', 'w', encoding='utf-8') as f:
        for name, shape in shapes.items():
            f.write(f'{name}: {shape}\n')
    
    return pred[0], prob[0]

def main():
    # 加载模型
    model = load_model()
    
    # 进行推理
    image_path = 'test_images/0.bmp'  # 替换为实际的测试图像路径
    pred, prob = inference_and_save(model, image_path)
    
    print(f'Predicted digit: {pred}')
    print('Probabilities:', prob)
    print('\nAll intermediate results have been saved to test_data directory')

if __name__ == '__main__':
    main() 