import numpy as np
from PIL import Image
import os

def convert_bmp_to_dat(bmp_path, dat_path, normalize=True, quantize=True):
    """将BMP图像转换为HLS可用的dat格式"""
    # 读取BMP图像
    image = Image.open(bmp_path)
    
    # 确保是灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整图像大小为32x32
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    image_np = np.array(image, dtype=np.float32)
    
    # 归一化到0-1
    if normalize:
        image_np = image_np / 255.0
    
    # 量化为16位定点数（8位整数部分，8位小数部分）
    if quantize:
        bits = 16
        int_bits = 8
        scale = 2.0 ** (bits - int_bits - 1)
        image_np = np.clip(np.round(image_np * scale), -2**(bits-1), 2**(bits-1)-1) / scale
    
    # 保存为逗号分隔的文本格式
    with open(dat_path, 'w', encoding='utf-8') as f:
        # 将数组展平并转换为字符串
        data_str = ','.join(f"{x:.10f}" for x in image_np.flatten())
        f.write(data_str)
    
    print(f"Converted {bmp_path} to {dat_path}")
    print(f"Shape: {image_np.shape}")
    print(f"Value range: [{image_np.min()}, {image_np.max()}]")
    return image_np.shape

def convert_all_test_images(input_dir='test_images', output_dir='test_data_hls'):
    """转换所有测试图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录所有形状信息
    shapes = {}
    
    # 转换每个测试图像
    for i in range(10):
        bmp_path = os.path.join(input_dir, f'{i}.bmp')
        if os.path.exists(bmp_path):
            dat_path = os.path.join(output_dir, f'input_{i}.dat')
            shape = convert_bmp_to_dat(bmp_path, dat_path)
            shapes[f'input_{i}'] = shape
    
    # 保存形状信息
    with open(os.path.join(output_dir, 'input_shapes.txt'), 'w', encoding='utf-8') as f:
        for name, shape in shapes.items():
            f.write(f'{name}: {shape}\n')

def read_dat_file(dat_path):
    """读取dat文件并验证内容"""
    with open(dat_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return np.array([float(x) for x in content.split(',')])

def verify_conversion(bmp_path, dat_path, shape):
    """验证转换是否正确"""
    # 读取原始图像
    image = Image.open(bmp_path)
    if image.mode != 'L':
        image = image.convert('L')
    # 调整图像大小为32x32
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    original = np.array(image, dtype=np.float32) / 255.0
    
    # 读取dat文件
    converted = read_dat_file(dat_path).reshape(shape)
    
    # 计算差异
    diff = np.abs(original - converted)
    print(f"\nVerification results for {bmp_path}:")
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")

if __name__ == '__main__':
    # 转换所有测试图像
    print("Converting test images to dat format...")
    convert_all_test_images()
    
    # 验证转换结果
    print("\nVerifying conversions...")
    for i in range(10):
        bmp_path = f'test_images/{i}.bmp'
        dat_path = f'test_data_hls/input_{i}.dat'
        if os.path.exists(bmp_path) and os.path.exists(dat_path):
            verify_conversion(bmp_path, dat_path, (32, 32)) 