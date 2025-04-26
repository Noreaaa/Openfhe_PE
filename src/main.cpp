#define PROFILE
#include "cmdline.h"
#include "openfhe.h"
#include "utils/types.hpp"
#include "encryption/encryption.hpp"
#include "cnn/model.hpp"
#include "cnn/conv2d.hpp"
#include "cnn/activation.hpp"
#include "cnn/pool.hpp"
#include <chrono>


#include <iostream>
#include <vector>

using namespace lbcrypto;
using std::vector;
CCParams<CryptoContextCKKSRNS> parameters; 

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
    usint dcrtBits               = 50;
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
    std::vector<uint32_t> levelBudget = {3, 3};

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
    uint32_t levelsAvailableAfterBootstrap = 11;
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
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

    parser.parse_check(argc, argv);
    int top = parser.get<int>("top");
    int bottom = parser.get<int>("bottom");
    int left = parser.get<int>("left");
    int right = parser.get<int>("right");
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

    // Step 2: Precomputations for bootstrapping
    cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);

    
    // Step 3: Key Generation
    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    // Generate bootstrapping keys.
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    
    std::vector<int32_t> rotate_index;

    // Test Step 1: 
    // generate random 3d vector as input
    int test_height = 32;
    int test_width = 32;
    for (int32_t i = 0; i < test_width; i++){
        rotate_index.push_back(i);
    }
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_index);
    types::double3d image_3d(3, types::double2d(test_height, 
        std::vector<double>(test_width, 0)));

    // create test image
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < test_height; j++){
            for (int k = 0; k < test_width; k++){
                image_3d[i][j][k] = rand() % 5;
            }
        }
    }

    // create the test filter
    int filter_height = 3;
    int filter_width = 3;
    int filter_num_1 = 32;
    int filter_num_2 = 64;
    int filter_num_3 = 128;
    types::double3d filter_3d_1(filter_num_1, types::double2d(filter_height,
    std::vector<double>(filter_width, 0)));

    for (int i = 0; i < filter_num_1; i++){
        for (int j = 0; j < filter_height; j++){
            for (int k = 0; k < filter_width; k++){
                filter_3d_1[i][j][k] = rand() % 5;
            }
        }
    }

    types::double3d filter_3d_2(filter_num_1, types::double2d(filter_height,
    std::vector<double>(filter_width, 0)));


    for (int i = 0; i < filter_num_1; i++){
        for (int j = 0; j < filter_height; j++){
            for (int k = 0; k < filter_width; k++){
                filter_3d_2[i][j][k] = rand() % 5;
            }
        }
    }

    types::double3d filter_3d_3(filter_num_3, types::double2d(filter_height,
        std::vector<double>(filter_width, 0)));

    for (int i = 0; i < filter_num_3; i++){
        for (int j = 0; j < filter_height; j++){
            for (int k = 0; k < filter_width; k++){
                filter_3d_3[i][j][k] = rand() % 5;
            }
        }
    }

   
    
    // initialize the bias
    std::vector<double> bias_1(filter_num_1, 0);
    std::vector<double> bias_2(filter_num_2, 0);
    std::vector<double> bias_3(filter_num_3, 0);

    int channel_size = 3;
    int height_start, height_end, width_start, width_end;


    Ciphertext<DCRTPoly> x_ctxt;
    std::cout << "image 3d:" << std::endl;
    print_3d(image_3d);

    height_start = 0;
    height_end = 1;
    width_start = 0;
    width_end = 1;

    ENCRYPTED_HEIGHT_START = height_start;
    ENCRYPTED_HEIGHT_END = height_end;
    ENCRYPTED_WIDTH_START = width_start;
    ENCRYPTED_WIDTH_END = width_end;
    std::vector<Ciphertext<DCRTPoly>> x_ctxt_vec;


    types::vector2d<Ciphertext<DCRTPoly>> x_ctxt_2d;
    Encrypt_MCSR_P(image_3d, numSlots, depth, cryptoContext, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);


    //return 0;
    std::cout << "finished encryption" << std::endl;

    std::cout << "create the model:" << std::endl;
    // create the model
    Network model;
    //model.add_layer(std::make_shared<Conv2d>(CONV_2D, "conv1", filter_3d, bias, 1, 0, numSlots));
    model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv1", filter_3d_1, bias_1, 1, 1, numSlots, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square1")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool1"), 2, 2, 0, numSlots));

    model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv2", filter_3d_2, bias_2, 1, 1, numSlots, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square2")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool2"), 2, 2, 0, numSlots));

    model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv3", filter_3d_3, bias_3, 1, 1, numSlots, AVG_POOLING));
    model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square3")));
    model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool3"), 2, 2, 0, numSlots));

    std::cout << "start prediction" << std::endl;
    CURRENT_HEIGHT = test_height;
    CURRENT_WIDTH = test_width;
    CURRENT_CHANNEL = 3;
    CRYPTOCONTEXT = cryptoContext;
    KEYPAIR = keyPair;

    auto start = std::chrono::high_resolution_clock::now();
    model.predict_P(x_ctxt_2d, image_3d);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return 0;
    

}