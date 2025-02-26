#include "cnn_accelerator.h"

// ReLU function
data_t relu(data_t x) {
    return (x > ZERO) ? x : ZERO;
}

// Optimized conv3x3 function
template<int H, int W>
void conv3x3(
    data_t feature_map[H][W],
    weight_t weight[KERNEL_SIZE][KERNEL_SIZE],
    data_t output_map[H][W]
) {
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=0
    
    // Adding padding
    data_t padded_map[H+2][W+2] = {ZERO}; // Initialize with zeros
    #pragma HLS ARRAY_PARTITION variable=padded_map cyclic factor=3 dim=2
    
    // Zero padding
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            padded_map[i+1][j+1] = feature_map[i][j];
        }
    }
    
    // Convolution calculation
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            data_t sum = ZERO;
            for(int ki = 0; ki < KERNEL_SIZE; ki++) {
                for(int kj = 0; kj < KERNEL_SIZE; kj++) {
                    #pragma HLS UNROLL
                    sum += padded_map[i+ki][j+kj] * weight[ki][kj];
                }
            }
            output_map[i][j] = sum;
        }
    }
}

// Optimized max_pool function
template<int H, int W>
void max_pool(
    data_t input_map[H][W],
    data_t output_map[H/2][W/2]
) {
    for(int i = 0; i < H/2; i++) {
        for(int j = 0; j < W/2; j++) {
            #pragma HLS PIPELINE II=1
            data_t max_val = input_map[i*2][j*2];
            max_val = (input_map[i*2][j*2+1] > max_val) ? input_map[i*2][j*2+1] : max_val;
            max_val = (input_map[i*2+1][j*2] > max_val) ? input_map[i*2+1][j*2] : max_val;
            max_val = (input_map[i*2+1][j*2+1] > max_val) ? input_map[i*2+1][j*2+1] : max_val;
            output_map[i][j] = max_val;
        }
    }
}

// Main function
void cnn_accelerator(
    data_t input_img[IMG_HEIGHT][IMG_WIDTH],
    data_t output_result[FC2_OUT],
    weight_t conv1_weight[CONV1_OUT_CH][KERNEL_SIZE][KERNEL_SIZE],
    weight_t conv2_weight[CONV2_OUT_CH][CONV1_OUT_CH][KERNEL_SIZE][KERNEL_SIZE],
    weight_t fc1_weight[FC1_OUT][CONV2_OUT_CH*8*8],
    weight_t fc2_weight[FC2_OUT][FC1_OUT]
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=input_img bundle=control
    #pragma HLS INTERFACE s_axilite port=output_result bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_weight bundle=control
    
    // Intermediate result buffers
    static data_t conv1_out[CONV1_OUT_CH][IMG_HEIGHT][IMG_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=2 dim=1
    
    static data_t pool1_out[CONV1_OUT_CH][IMG_HEIGHT/2][IMG_WIDTH/2];
    #pragma HLS ARRAY_PARTITION variable=pool1_out cyclic factor=2 dim=1
    
    static data_t conv2_out[CONV2_OUT_CH][IMG_HEIGHT/2][IMG_WIDTH/2];
    #pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=2 dim=1
    
    static data_t pool2_out[CONV2_OUT_CH][IMG_HEIGHT/4][IMG_WIDTH/4];
    #pragma HLS ARRAY_PARTITION variable=pool2_out cyclic factor=2 dim=1
    
    static data_t fc1_out[FC1_OUT];
    #pragma HLS ARRAY_PARTITION variable=fc1_out cyclic factor=2 dim=1
    
    static data_t fc2_out[FC2_OUT];
    #pragma HLS ARRAY_PARTITION variable=fc2_out complete dim=1

    // First convolution layer
    for(int oc = 0; oc < CONV1_OUT_CH; oc++) {
        conv3x3<IMG_HEIGHT, IMG_WIDTH>(input_img, conv1_weight[oc], conv1_out[oc]);
        // ReLU
        for(int i = 0; i < IMG_HEIGHT; i++) {
            for(int j = 0; j < IMG_WIDTH; j++) {
                #pragma HLS PIPELINE II=1
                conv1_out[oc][i][j] = relu(conv1_out[oc][i][j]);
            }
        }
    }
    
    // First pooling layer
    for(int oc = 0; oc < CONV1_OUT_CH; oc++) {
        max_pool<IMG_HEIGHT, IMG_WIDTH>(conv1_out[oc], pool1_out[oc]);
    }
    
    // Second convolution layer
    for(int oc = 0; oc < CONV2_OUT_CH; oc++) {
        data_t temp_out[IMG_HEIGHT/2][IMG_WIDTH/2] = {ZERO};
        for(int ic = 0; ic < CONV1_OUT_CH; ic++) {
            data_t conv_temp[IMG_HEIGHT/2][IMG_WIDTH/2];
            conv3x3<IMG_HEIGHT/2, IMG_WIDTH/2>(pool1_out[ic], conv2_weight[oc][ic], conv_temp);
            // Accumulate results
            for(int i = 0; i < IMG_HEIGHT/2; i++) {
                for(int j = 0; j < IMG_WIDTH/2; j++) {
                    #pragma HLS PIPELINE II=1
                    temp_out[i][j] += conv_temp[i][j];
                }
            }
        }
        // ReLU
        for(int i = 0; i < IMG_HEIGHT/2; i++) {
            for(int j = 0; j < IMG_WIDTH/2; j++) {
                #pragma HLS PIPELINE II=1
                conv2_out[oc][i][j] = relu(temp_out[i][j]);
            }
        }
    }
    
    // Second pooling layer
    for(int oc = 0; oc < CONV2_OUT_CH; oc++) {
        max_pool<IMG_HEIGHT/2, IMG_WIDTH/2>(conv2_out[oc], pool2_out[oc]);
    }
    
    // Flatten data
    data_t flatten[CONV2_OUT_CH*8*8];
    int idx = 0;
    for(int c = 0; c < CONV2_OUT_CH; c++) {
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < 8; j++) {
                #pragma HLS PIPELINE II=1
                flatten[idx++] = pool2_out[c][i][j];
            }
        }
    }
    
    // First fully connected layer
    for(int i = 0; i < FC1_OUT; i++) {
        data_t sum = ZERO;
        for(int j = 0; j < CONV2_OUT_CH*8*8; j++) {
            #pragma HLS PIPELINE II=1
            sum += flatten[j] * fc1_weight[i][j];
        }
        fc1_out[i] = relu(sum);
    }
    
    // Second fully connected layer
    for(int i = 0; i < FC2_OUT; i++) {
        data_t sum = ZERO;
        for(int j = 0; j < FC1_OUT; j++) {
            #pragma HLS PIPELINE II=1
            sum += fc1_out[j] * fc2_weight[i][j];
        }
        fc2_out[i] = sum;
    }
    
    // Output results
    for(int i = 0; i < FC2_OUT; i++) {
        #pragma HLS PIPELINE II=1
        output_result[i] = fc2_out[i];
    }
} 