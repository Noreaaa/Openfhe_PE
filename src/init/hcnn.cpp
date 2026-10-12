#include "hcnn.hpp"
#include "../cnn/conv2d.hpp"
#include "../cnn/activation.hpp"
#include "../cnn/pool.hpp"
#include "../cnn/linear.hpp"
#include "../encryption/encryption.hpp"
#include "../utils/dataloader.hpp"
#include "../utils/helper.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace lbcrypto;

void InitHCNN(Network& model, const std::string& hcnn_path, uint32_t numSlots, double eps) {



        types::double4d filter_4d_1, filter_4d_2, filter_4d_3;
        std::vector<double> gamma_1, gamma_2, gamma_3;
        std::vector<double> beta_1, beta_2, beta_3;
        std::vector<double> mean_1, mean_2, mean_3;
        std::vector<double> var_1, var_2, var_3;
        types::double2d linear_weight_1, linear_weight_2;
        std::vector<double> linear_bias_1, linear_bias_2;  
        std::vector<double> bias_1, bias_2, bias_3;

        std::cout << "HCNN model is being loaded..." << std::endl;

        filter_4d_1 = LoadConv2dWeight(hcnn_path + "conv1.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv1.0.bias.npy", bias_1);
        LoadConv2dBias(hcnn_path + "conv1.1.weight.npy", gamma_1);
        LoadConv2dBias(hcnn_path + "conv1.1.bias.npy", beta_1);
        LoadConv2dBias(hcnn_path + "conv1.1.running_mean.npy", mean_1);
        LoadConv2dBias(hcnn_path + "conv1.1.running_var.npy", var_1);


        filter_4d_2 = LoadConv2dWeight(hcnn_path + "conv2.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv2.0.bias.npy", bias_2);
        LoadConv2dBias(hcnn_path + "conv2.1.weight.npy", gamma_2);
        LoadConv2dBias(hcnn_path + "conv2.1.bias.npy", beta_2);
        LoadConv2dBias(hcnn_path + "conv2.1.running_mean.npy", mean_2);
        LoadConv2dBias(hcnn_path + "conv2.1.running_var.npy", var_2);

        filter_4d_3 = LoadConv2dWeight(hcnn_path + "conv3.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv3.0.bias.npy", bias_3);
        LoadConv2dBias(hcnn_path + "conv3.1.weight.npy", gamma_3);
        LoadConv2dBias(hcnn_path + "conv3.1.bias.npy", beta_3);
        LoadConv2dBias(hcnn_path + "conv3.1.running_mean.npy", mean_3);
        LoadConv2dBias(hcnn_path + "conv3.1.running_var.npy", var_3);
        
        LoadLinearWeight(hcnn_path + "fc1.weight.npy", linear_weight_1);
        LoadLinearWeight(hcnn_path + "fc2.weight.npy", linear_weight_2);
        LoadConv2dBias(hcnn_path + "fc1.bias.npy", linear_bias_1);
        LoadConv2dBias(hcnn_path + "fc2.bias.npy", linear_bias_2);

        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv1_bn1", filter_4d_1, 1, 1, numSlots, gamma_1, beta_1, mean_1, var_1, eps, bias_1, AVG_POOLING));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool1"), 2, 2, 0, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv2_bn2", filter_4d_2, 1, 1, numSlots, gamma_2, beta_2, mean_2, var_2, eps, bias_2, AVG_POOLING));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool2"), 2, 2, 0, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv3_bn3", filter_4d_3, 1, 1, numSlots, gamma_3, beta_3, mean_3, var_3, eps, bias_3, AVG_POOLING));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool3"), 2, 2, 0, numSlots));
        model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear1", linear_weight_1, linear_bias_1, numSlots));
        model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear2", linear_weight_2, linear_bias_2, numSlots));
    
}
