#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

// 使用相对路径包含Xilinx HLS头文件
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

// 定义数据类型
typedef ap_fixed<16,8,AP_RND,AP_SAT> data_t;
typedef ap_fixed<16,8,AP_RND,AP_SAT> weight_t;

// 定义常量
const data_t ZERO = 0;

// 网络参数定义
const int IMG_HEIGHT = 32;
const int IMG_WIDTH = 32;
const int CONV1_OUT_CH = 16;
const int CONV2_OUT_CH = 32;
const int FC1_OUT = 128;
const int FC2_OUT = 10;
const int KERNEL_SIZE = 3;

// 权重数组声明
extern "C" {
    extern weight_t conv1_weight[KERNEL_SIZE][KERNEL_SIZE][CONV1_OUT_CH];
    extern weight_t conv2_weight[KERNEL_SIZE][KERNEL_SIZE][CONV1_OUT_CH][CONV2_OUT_CH];
    extern weight_t fc1_weight[CONV2_OUT_CH*8*8][FC1_OUT];
    extern weight_t fc2_weight[FC1_OUT][FC2_OUT];
}

// 函数声明
extern "C" {
    void cnn_accelerator(
        data_t input_img[IMG_HEIGHT][IMG_WIDTH],
        data_t output_result[FC2_OUT]
    );
}

#endif 