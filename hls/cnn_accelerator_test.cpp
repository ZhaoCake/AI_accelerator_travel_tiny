#include "cnn_accelerator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// 用于加载和验证数据的辅助函数
void load_data(const std::string& filename, float* data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data), size * sizeof(float));
    file.close();
}

void compare_results(const std::string& golden_file, data_t* result, size_t size, const std::string& layer_name) {
    std::vector<float> golden_data(size);
    load_data(golden_file, golden_data.data(), size);
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(golden_data[i] - result[i].to_float());
        avg_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    avg_diff /= size;
    
    std::cout << "\nResults for " << layer_name << ":" << std::endl;
    std::cout << "Max difference: " << max_diff << " at index " << max_diff_idx << std::endl;
    std::cout << "Average difference: " << avg_diff << std::endl;
}

int main() {
    // 测试数据路径
    const std::string test_dir = "testbench/digit_0/";
    
    // 分配内存
    data_t input_img[IMG_HEIGHT][IMG_WIDTH];
    data_t output_result[FC2_OUT];
    
    // 加载输入数据
    std::cout << "Loading input data..." << std::endl;
    std::ifstream input_file(test_dir + "input.dat", std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error: Cannot open input file" << std::endl;
        return 1;
    }
    input_file.read(reinterpret_cast<char*>(input_img), sizeof(input_img));
    input_file.close();
    
    // 打印输入数据的一些统计信息
    float min_val = input_img[0][0].to_float();
    float max_val = min_val;
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            float val = input_img[i][j].to_float();
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }
    std::cout << "Input range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // 运行推理
    std::cout << "\nRunning inference..." << std::endl;
    cnn_accelerator(input_img, output_result);
    
    // 打印并验证结果
    std::cout << "\nPrediction results:" << std::endl;
    float max_prob = output_result[0].to_float();
    int pred_digit = 0;
    
    for (int i = 0; i < FC2_OUT; i++) {
        float prob = output_result[i].to_float();
        std::cout << "Class " << i << ": " << std::fixed << std::setprecision(6) << prob << std::endl;
        if (prob > max_prob) {
            max_prob = prob;
            pred_digit = i;
        }
    }
    
    std::cout << "\nPredicted digit: " << pred_digit << std::endl;
    
    // 与golden数据比较
    std::cout << "\nComparing with golden data..." << std::endl;
    compare_results(test_dir + "fc2_output.dat", output_result, FC2_OUT, "Final Output");
    
    return 0;
} 