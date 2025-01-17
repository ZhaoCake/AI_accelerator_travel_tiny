import torch
from torchvision import datasets, transforms
from PIL import Image
import os
from net_code import SimpleCNN
from inference import inference_and_save
import numpy as np

def get_mnist_samples():
    """从MNIST测试集中获取每个数字的样本"""
    # 创建保存目录
    os.makedirs('test_images', exist_ok=True)
    
    # 加载MNIST测试集
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 保持原始MNIST大小
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # 为每个数字找一个样本
    samples_found = set()
    for idx, (image, label) in enumerate(test_dataset):
        if label not in samples_found:
            # 将tensor转换回PIL图像并保存
            image_np = (image.squeeze().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_path = f'test_images/{label}.bmp'
            image_pil.save(image_path)
            print(f"Saved digit {label} to {image_path}")
            samples_found.add(label)
            
        if len(samples_found) == 10:  # 找到所有数字后退出
            break
    
    print("\nSaved test images for all digits (0-9)")

def test_model_on_samples(model_path='mnist_cnn.pth'):
    """测试模型在所有样本上的表现"""
    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\nTesting model on all samples:")
    correct = 0
    total = 0
    
    for i in range(10):
        image_path = f'test_images/{i}.bmp'
        if os.path.exists(image_path):
            pred, prob = inference_and_save(model, image_path, f'test_data/digit_{i}')
            print(f"\nInput digit: {i}")
            print(f"Predicted digit: {pred}")
            print("Probabilities:", [f"{p:.4f}" for p in prob])
            
            if pred == i:
                correct += 1
            total += 1
    
    print(f"\nAccuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")

if __name__ == '__main__':
    # 获取测试图像
    get_mnist_samples()
    
    # 在所有样本上测试模型
    test_model_on_samples() 