#define PROFILE
#include "cmdline.h"
#include "openfhe.h"
#include "utils/types.hpp"
#include "utils/dataloader.hpp"
#include "encryption/encryption.hpp"
#include "cnn/model.hpp"
#include "cnn/conv2d.hpp"
#include "cnn/activation.hpp"
#include "cnn/pool.hpp"
#include "cnn/bootstrap.hpp"
#include "cnn/linear.hpp"
#include <chrono>


#include <iostream>
#include <vector>

using namespace lbcrypto;
using std::vector;
CCParams<CryptoContextCKKSRNS> parameters; 

std::string data_path = "../python_model/saved_models/parameters/";
std::string cifar_image_path = "../datasets/cifar-10/test_batch.bin";

void SetParam(uint32_t RingDim){
    // Step 1: Set CryptoContext

    // A. Specify main parameters
    /*  A1) Secret key distribution
    * The secret key distribution for CKKS should either be SPARSE_TERNARY or UNIFORM_TERNARY.
    * The SPARSE_TERNARY distribution was used in the original CKKS paper,
    * but in this example, we use UNIFORM_TERNARY because this is included in the homomorphic
    * encryption standard.
    */
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    /*  A2) Desired security level based on FHE standards.
    * In this example, we use the "NotSet" option, so the example can run more quickly with
    * a smaller ring dimension. Note that this should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices. In production-like environments, we recommend using
    * HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic for 128-bit, 192-bit,
    * or 256-bit security, respectively. If you choose one of these as your security level,
    * you do not need to set the ring dimension.
    */
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(RingDim);

    /*  A3) Key switching parameters.
    * By default, we use HYBRID key switching with a digit size of 3.
    * Choosing a larger digit size can reduce complexity, but the size of keys will increase.
    * Note that you can leave these lines of code out completely, since these are the default values.
    */
    parameters.SetNumLargeDigits(3);
    parameters.SetKeySwitchTechnique(HYBRID);

    /*  A4) Scaling parameters.
    * By default, we set the modulus sizes and rescaling technique to the following values
    * to obtain a good precision and performance tradeoff. We recommend keeping the parameters
    * below unless you are an FHE expert.
    */
#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    // Currently, only FIXEDMANUAL and FIXEDAUTO modes are supported for 128-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint dcrtBits               = 78;
    usint firstMod               = 89;
#else
    // All modes are supported for 64-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 59;
    usint firstMod               = 60;
#endif

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    /*  A4) Bootstrapping parameters.
    * We set a budget for the number of levels we can consume in bootstrapping for encoding and decoding, respectively.
    * Using larger numbers of levels reduces the complexity and number of rotation keys,
    * but increases the depth required for bootstrapping.
	* We must choose values smaller than ceil(log2(slots)). A level budget of {4, 4} is good for higher ring
    * dimensions (65536 and higher).
    */
    std::vector<uint32_t> levelBudget = {4, 4};

    /* We give the user the option of configuring values for an optimization algorithm in bootstrapping.
    * Here, we specify the giant step for the baby-step-giant-step algorithm in linear transforms
    * for encoding and decoding, respectively. Either choose this to be a power of 2
    * or an exact divisor of the number of slots. Setting it to have the default value of {0, 0} allows OpenFHE to choose
    * the values automatically.
    */
    std::vector<uint32_t> bsgsDim = {0, 0};

     /*  A5) Multiplicative depth.
    * The goal of bootstrapping is to increase the number of available levels we have, or in other words,
    * to dynamically increase the multiplicative depth. However, the bootstrapping procedure itself
    * needs to consume a few levels to run. We compute the number of bootstrapping levels required
    * using GetBootstrapDepth, and add it to levelsAvailableAfterBootstrap to set our initial multiplicative
    * depth.
    */
    uint32_t levelsAvailableAfterBootstrap = 10;
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    std::cout << "depth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);
}

void test_ablation(){
    for (int i = 5; i < 13; i++){
        SetParam(1 << i);
        CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
        cryptoContext->Enable(PKE);
        cryptoContext->Enable(KEYSWITCH);
        cryptoContext->Enable(LEVELEDSHE);
        cryptoContext->Enable(ADVANCEDSHE);
        cryptoContext->Enable(FHE);
        
    
        int ringDim = cryptoContext->GetRingDimension();
        std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;
        auto keyPair = cryptoContext->KeyGen();
        cryptoContext->EvalMultKeyGen(keyPair.secretKey);
        cryptoContext->EvalSumKeyGen(keyPair.secretKey);

        std::vector<double> vec(ringDim/2);
        for (int i = 0; i < ringDim/2; i++){
            vec[i] = rand() % 10;
        }
        Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(vec);
        Ciphertext<DCRTPoly> ctxt = cryptoContext->Encrypt(keyPair.publicKey, ptxt);
        auto start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalAdd(ctxt, ptxt);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Add in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalMult(ctxt, ptxt);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Mult in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalSum(ctxt, ringDim/2);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Sum in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
}

int main(int argc, char *argv[]) {
    cmdline::parser parser;
    parser.add<int>("top", 0, "top of the encryption region 0-31", false, 0);
    parser.add<int>("bottom", 0, "bottom of the encryption region 0-31", false, 0);
    parser.add<int>("left", 0, "left of the encryption region 0-31", false, 0);
    parser.add<int>("right", 0, "right of the encryption region 0-31", false, 0);
    parser.add<int>("nums", 0, "number of images to be tested", false, 0);

    parser.parse_check(argc, argv);
    int height_start = parser.get<int>("top");
    int height_end = parser.get<int>("bottom");
    int width_start = parser.get<int>("left");
    int width_end = parser.get<int>("right");
    int test_nums = parser.get<int>("nums");



    //test_ablation();
    //return 0;
    int ringDim = 64;
    SetParam(ringDim);
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    uint32_t numSlots = ringDim / 2;
    int depth = 10;

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    // Enable features that you wish to use. Note, we must enable FHE to use bootstrapping.
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);
    

    ringDim = cryptoContext->GetRingDimension();
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    // Generate bootstrapping keys.
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    
    std::vector<int32_t> rotate_index;

    // create the rotate key
    int image_size = 32;

    for (int32_t i = 0; i < image_size; i++){
        rotate_index.push_back(i);
    }
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_index);
    

    // create the test filter
    int filter_height = 3;
    int filter_width = 3;
    int filter_num_1 = 3;
    int filter_num_2 = 64;
    int filter_num_3 = 128;

    std::vector<double> bias_1(filter_num_1, 0);
    std::vector<double> bias_2(filter_num_2, 0);
    std::vector<double> bias_3(filter_num_3, 0);


    Ciphertext<DCRTPoly> x_ctxt;


    std::vector<Ciphertext<DCRTPoly>> x_ctxt_vec;


    types::vector2d<Ciphertext<DCRTPoly>> x_ctxt_2d;





    std::cout << "create the model:" << std::endl;
    // create the model
    Network model;

    std::vector<double> gamma_1;
    std::vector<double> beta_1;
    std::vector<double> mean_1;
    std::vector<double> var_1;
    std::vector<double> epsilon_1(32, 1e-5);
    std::vector<double> gamma_2;
    std::vector<double> beta_2;
    std::vector<double> mean_2;
    std::vector<double> var_2;
    std::vector<double> gamma_3;
    std::vector<double> beta_3;
    std::vector<double> mean_3;
    std::vector<double> var_3;
    types::double2d linear_weight_1;
    types::double2d linear_weight_2;
    std::vector<double> linear_bias_1;  
    std::vector<double> linear_bias_2;


    types::double4d filter_4d_1 = LoadConv2dWeight(data_path + "conv1.0.weight.npy");
    LoadConv2dBias(data_path + "conv1.0.bias.npy", bias_1);
    LoadConv2dBias(data_path + "conv1.1.weight.npy", gamma_1);
    LoadConv2dBias(data_path + "conv1.1.bias.npy", beta_1);
    LoadConv2dBias(data_path + "conv1.1.running_mean.npy", mean_1);
    LoadConv2dBias(data_path + "conv1.1.running_var.npy", var_1);


    types::double4d filter_4d_2 = LoadConv2dWeight(data_path + "conv2.0.weight.npy");
    LoadConv2dBias(data_path + "conv2.0.bias.npy", bias_2);
    LoadConv2dBias(data_path + "conv2.1.weight.npy", gamma_2);
    LoadConv2dBias(data_path + "conv2.1.bias.npy", beta_2);
    LoadConv2dBias(data_path + "conv2.1.running_mean.npy", mean_2);
    LoadConv2dBias(data_path + "conv2.1.running_var.npy", var_2);

    types::double4d filter_4d_3 = LoadConv2dWeight(data_path + "conv3.0.weight.npy");
    LoadConv2dBias(data_path + "conv3.0.bias.npy", bias_3);
    LoadConv2dBias(data_path + "conv3.1.weight.npy", gamma_3);
    LoadConv2dBias(data_path + "conv3.1.bias.npy", beta_3);
    LoadConv2dBias(data_path + "conv3.1.running_mean.npy", mean_3);
    LoadConv2dBias(data_path + "conv3.1.running_var.npy", var_3);
    
    LoadLinearWeight(data_path + "fc1.weight.npy", linear_weight_1);
    LoadLinearWeight(data_path + "fc2.weight.npy", linear_weight_2);
    LoadConv2dBias(data_path + "fc1.bias.npy", linear_bias_1);
    LoadConv2dBias(data_path + "fc2.bias.npy", linear_bias_2);

    

    
    //model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv1", filter_3d_1, bias_1, 1, 1, numSlots, CONV_2D));
    model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv1_bn", filter_4d_1, 1, 1, numSlots, gamma_1, beta_1, mean_1, var_1, epsilon_1, bias_1, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square1")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool1"), 2, 2, 0, numSlots));
    //model.add_layer(std::make_shared<Bootstrap_P>(BOOTSTRAP, std::string("bootstrap1")));
    //model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv2", filter_3d_2, bias_2, 1, 1, numSlots, AVG_POOLING));
    model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv2_bn", filter_4d_2, 1, 1, numSlots, gamma_2, beta_2, mean_2, var_2, epsilon_1, bias_2, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square2")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool2"), 2, 2, 0, numSlots));

    //model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv3", filter_3d_3, bias_3, 1, 1, numSlots, AVG_POOLING));
    model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv2_bn", filter_4d_3, 1, 1, numSlots, gamma_3, beta_3, mean_3, var_3, epsilon_1, bias_3, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square3")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool3"), 2, 2, 0, numSlots));

    model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear1", linear_weight_1, linear_bias_1, numSlots));
    model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear2", linear_weight_2, linear_bias_2, numSlots));



    types::double3d image_3d(3, types::double2d(image_size, 
        std::vector<double>(image_size, 0)));
    types::double3d golden_output(3, types::double2d(image_size, 
        std::vector<double>(image_size, 0)));
    int label = 0;
    long long total_time = 0;
    int correct_count = 0; 
    CRYPTOCONTEXT = cryptoContext;
    for (int i = 0; i < test_nums; i++){
        LoadImageCifar(cifar_image_path, image_3d, label, i);
        LoadImageCifar(cifar_image_path, golden_output, label, i);
        Encrypt_MCSR_P(image_3d, numSlots, depth, cryptoContext, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);

        KEYPAIR = keyPair;
        ENCRYPTED_HEIGHT_START = height_start;
        ENCRYPTED_HEIGHT_END = height_end;
        ENCRYPTED_WIDTH_START = width_start;
        ENCRYPTED_WIDTH_END = width_end;
        std::cout << "image[" << i <<"]:" << std::endl;
        //print_3d(image_3d);


        auto start = std::chrono::high_resolution_clock::now();
        //std::cout << "check level at start: " << x_ctxt_2d[0][0]->GetLevel() << std::endl;
        int predict_label = model.predict_P(x_ctxt_2d, image_3d);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "true result: " << label << std::endl;

        if (predict_label == label){
            correct_count++;
        }

        //GoldenConv2d(golden_output, filter_4d_1, bias_1, 1, 1);
        //GoldenBN(golden_output, gamma_1, beta_1, mean_1, var_1, epsilon_1);
        //golden_Square(golden_output);
        //golden_AvgPooling(golden_output, 2, 2);
        //GoldenConv2d(golden_output, filter_4d_2, bias_2, 1, 1);
        //GoldenBN(golden_output, gamma_2, beta_2, mean_2, var_2, epsilon_1);
        //golden_Square(golden_output);
        //golden_AvgPooling(golden_output, 2, 2);
        //GoldenConv2d(golden_output, filter_4d_3, bias_2, 1, 1);
        //GoldenBN(golden_output, gamma_3, beta_3, mean_3, var_3, epsilon_1);
        //golden_Square(golden_output);
        //golden_AvgPooling(golden_output, 2, 2);

        //std::vector<double> golden_output_1d;
        //GoldenLinear_3d_input(golden_output, linear_weight_1, linear_bias_1, golden_output_1d);
        //GoldenLinear(golden_output_1d, linear_weight_2, linear_bias_2);

        //for (int i = 0; i < static_cast<int>(golden_output_1d.size()); i++){
        //    std::cout << "golden_output_1d[" << i << "]: " << golden_output_1d[i] << std::endl;
        //}

    }

    std::cout << "average predicted time: " << total_time/test_nums << "ms" << std::endl;
    double accuracy = (double)correct_count / (double)test_nums;
    std::cout << "accuracy: " << accuracy * 100 << "%" << std::endl;






    return 0;
    

}