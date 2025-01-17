import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net_code import SimpleCNN
import numpy as np
import os

def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # 测试模型
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.2f}%)')

def save_weights_for_hls(model, save_dir='weights'):
    """将模型权重保存为HLS可用的格式"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 量化参数
    bits = 16
    int_bits = 8
    scale = 2.0 ** (bits - int_bits - 1)
    
    def quantize(data):
        return np.clip(np.round(data * scale), -2**(bits-1), 2**(bits-1)-1) / scale
    
    def save_array_to_dat(arr, filename):
        """将数组保存为逗号分隔的文本格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            # 将数组展平并转换为字符串
            data_str = ','.join(f"{x:.10f}" for x in arr.flatten())
            # 添加花括号包裹
            f.write(data_str)
    
    # 保存conv1权重 (NCHW -> HWC)
    conv1_weight = model.conv1.weight.data.numpy()  # 原始形状(16,1,3,3)
    conv1_weight = quantize(conv1_weight)
    conv1_weight = conv1_weight.squeeze(1)  # 移除输入通道维度
    conv1_weight = np.transpose(conv1_weight, (1,2,0))  # 变为(3,3,16)
    save_array_to_dat(conv1_weight, f'{save_dir}/conv1_weight.dat')
    
    # 保存conv2权重 (NCHW -> HWIO)
    conv2_weight = model.conv2.weight.data.numpy()  # 原始形状(32,16,3,3)
    conv2_weight = quantize(conv2_weight)
    conv2_weight = np.transpose(conv2_weight, (2,3,1,0))  # 变为(3,3,16,32)
    save_array_to_dat(conv2_weight, f'{save_dir}/conv2_weight.dat')
    
    # 保存fc1权重
    fc1_weight = model.fc1.weight.data.numpy()  # 原始形状(128, 2048)
    fc1_weight = quantize(fc1_weight)
    fc1_weight = fc1_weight.T  # 转置为(2048, 128)
    save_array_to_dat(fc1_weight, f'{save_dir}/fc1_weight.dat')
    
    # 保存fc2权重
    fc2_weight = model.fc2.weight.data.numpy()  # 原始形状(10, 128)
    fc2_weight = quantize(fc2_weight)
    fc2_weight = fc2_weight.T  # 转置为(128, 10)
    save_array_to_dat(fc2_weight, f'{save_dir}/fc2_weight.dat')
    
    # 保存权重形状信息
    shapes = {
        'conv1': (3,3,16),  # HWC
        'conv2': (3,3,16,32),  # HWIO
        'fc1': (2048,128),  # 转置后的形状
        'fc2': (128,10)     # 转置后的形状
    }
    with open(f'{save_dir}/shapes.txt', 'w') as f:
        for name, shape in shapes.items():
            f.write(f'{name}: {shape}\n')

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 准备MNIST数据集
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小为32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 创建和训练模型
    model = SimpleCNN()
    train_model(model, train_loader, test_loader)
    
    # 保存模型权重
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    
    # 保存为HLS格式
    save_weights_for_hls(model)

if __name__ == '__main__':
    main() 