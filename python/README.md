# Python工作目录

本目录包含了MNIST CNN模型的训练、测试和数据准备相关的Python代码。

## 主要工作流程

1. 模型训练与保存
   - 使用`train_model.py`训练CNN模型
   - 模型结构定义在`net_code.py`中
   - 训练后的模型保存为PyTorch格式(.pth)和HLS格式(.dat)

2. 测试数据准备
   - 使用`get_test_images.py`从MNIST测试集获取样本
   - 为每个数字(0-9)保存一个测试样本
   - 图像保存为BMP格式，便于查看和调试

3. 数据格式转换
   - 使用`convert_images_to_dat.py`将BMP图像转换为HLS可用格式
   - 支持图像预处理：
     * 调整大小为32x32
     * 灰度化处理
     * 归一化到[0,1]范围
     * 定点数量化(16位)
   - 数据以逗号分隔的文本格式保存

4. 模型推理与验证
   - 使用`inference.py`进行模型推理
   - 保存每一层的输出结果
   - 生成用于HLS验证的golden data

## 数据格式说明

1. 输入数据 (input.dat)
   - 尺寸：32x32
   - 格式：灰度图，归一化到[0,1]
   - 存储：逗号分隔的浮点数

2. 中间层输出
   - conv1_output.dat: 32x32x16
   - pool1_output.dat: 16x16x16
   - conv2_output.dat: 16x16x32
   - pool2_output.dat: 8x8x32
   - fc1_output.dat: 1x128
   - fc2_output.dat: 1x10

3. 权重文件
   - conv1_weight.dat: 3x3x16
   - conv2_weight.dat: 3x3x16x32
   - fc1_weight.dat: 2048x128
   - fc2_weight.dat: 128x10

## 目录结构
```
./
├── net_code.py              # CNN模型定义
├── train_model.py           # 模型训练代码
├── get_test_images.py       # 测试图像获取
├── convert_images_to_dat.py # 数据格式转换
├── inference.py             # 模型推理
├── test_images/            # BMP格式测试图像
├── test_data/              # HLS验证数据
└── weights/                # 模型权重
```

## 使用说明

1. 训练模型并保存权重：
```bash
python train_model.py
```

2. 获取测试图像并推理生成测试数据：
```bash
python get_test_images.py
```

另外两个脚本是辅助功能。

## 注意事项

1. 所有数据使用UTF-8编码保存
2. 浮点数保留10位小数
3. 确保目录结构完整再运行脚本
4. 建议按照上述顺序执行各个步骤


