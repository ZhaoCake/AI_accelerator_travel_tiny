# HLS CNN加速器实现

本目录包含使用Vivado HLS实现的CNN加速器代码。

## 文件说明

- `cnn_accelerator.h`: 头文件,定义了数据类型、网络参数和函数接口
- `cnn_accelerator.cpp`: 加速器主要实现代码,包含卷积、池化等操作
- `cnn_accelerator_test.cpp`: 测试代码,用于验证加速器功能正确性

## 数据来源

所有权重数据和测试数据来自Python训练的模型:

1. 权重数据
   - 由`python/train_model.py`训练得到模型权重
   - 通过`save_weights_for_hls()`函数转换为HLS可用的格式
   - 保存在`weights/`目录下

2. 测试数据
   - 由`python/inference.py`生成
   - 包含输入图像和各层的golden输出数据
   - 保存在`test_data/`目录下

## 数据格式

1. 权重文件
   - conv1_weight.dat: 3x3x16 (HWC格式)
   - conv2_weight.dat: 3x3x16x32 (HWIO格式) 
   - fc1_weight.dat: 2048x128
   - fc2_weight.dat: 128x10

2. 测试数据
   - input.dat: 32x32 输入图像
   - conv1_output.dat: 32x32x16 第一层卷积输出
   - pool1_output.dat: 16x16x16 第一层池化输出
   - conv2_output.dat: 16x16x32 第二层卷积输出
   - pool2_output.dat: 8x8x32 第二层池化输出
   - fc1_output.dat: 1x128 第一层全连接输出
   - fc2_output.dat: 1x10 最终输出

## 使用说明

1. 确保已生成所需的权重和测试数据
2. 使用Vivado HLS打开工程
3. 运行C仿真验证功能
4. 进行综合和实现

## 注意事项

1. 所有数据使用定点数格式(ap_fixed<16,8>)
2. 确保数据文件路径正确
3. 建议先进行C仿真再综合
