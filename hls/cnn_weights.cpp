#include "cnn_accelerator.h"

// 权重定义
weight_t conv1_weight[KERNEL_SIZE][KERNEL_SIZE][CONV1_OUT_CH] = {
    #include "weights/conv1_weight.dat"
};

weight_t conv2_weight[KERNEL_SIZE][KERNEL_SIZE][CONV1_OUT_CH][CONV2_OUT_CH] = {
    #include "weights/conv2_weight.dat"
};

weight_t fc1_weight[CONV2_OUT_CH*8*8][FC1_OUT] = {
    #include "weights/fc1_weight.dat"
};

weight_t fc2_weight[FC1_OUT][FC2_OUT] = {
    #include "weights/fc2_weight.dat"
}; 