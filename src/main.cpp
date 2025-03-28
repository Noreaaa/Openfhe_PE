#define PROFILE
#include "cmdline.h"
#include "openfhe.h"
#include "utils/types.hpp"
#include "encryption/encryption.hpp"
#include "cnn/model.hpp"
#include "cnn/conv2d.hpp"
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
    uint32_t levelsAvailableAfterBootstrap = 10;
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
    int ringDim = 1 << 8;
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
    int test_height = 6;
    int test_width = 6;
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
                image_3d[i][j][k] = rand() % 10;
            }
        }
    }

    // create the test filter
    int filter_height = 3;
    int filter_width = 3;
    int filter_num = 1;
    types::double3d filter_3d(filter_num, types::double2d(filter_height,
    std::vector<double>(filter_width, 0)));

    for (int i = 0; i < filter_num; i++){
        for (int j = 0; j < filter_height; j++){
            for (int k = 0; k < filter_width; k++){
                filter_3d[i][j][k] = rand() % 10;
            }
        }
    }

    std::cout << "filter 3d:" << std::endl;
    print_3d(filter_3d);

    // initialize the bias
    std::vector<double> bias(filter_num, 0);

    // Test step 2: doing partial encryption 

    // to do: calculate channel size to be used in single ciphertext
    //int enc_height;
    //int enc_width;
    // top bottom left right 
    //std::tuple<int, int, int, int> EncRegion = CalculateRegionHCNN(10, 20, 10, 20);
    int channel_size = 3;
    int height_start, height_end, width_start, width_end;

    //enc_height = std::get<1>(EncRegion) - std::get<0>(EncRegion);
    //enc_width = std::get<3>(EncRegion) - std::get<2>(EncRegion);

    Ciphertext<DCRTPoly> x_ctxt;
    std::cout << "image 3d:" << std::endl;
    print_3d(image_3d);

    //std::cout << "run partial encrypt" << std::endl;

    // to be changed 
    height_start = 2;
    height_end = 4;
    width_start = 2;
    width_end = 4;

    ENCRYPTED_HEIGHT_START = height_start;
    ENCRYPTED_HEIGHT_END = height_end;
    ENCRYPTED_WIDTH_START = width_start;
    ENCRYPTED_WIDTH_END = width_end;
    std::vector<Ciphertext<DCRTPoly>> x_ctxt_vec;
    /* 
    Partial_Encrypt(image_3d, numSlots, depth, cryptoContext, keyPair, 
        3, enc_height, enc_width, height_start, height_end, 
        width_start, width_end, x_ctxt);
    */
    /*
    Partial_Encrypt_Sparse(image_3d, numSlots, depth, cryptoContext, keyPair,
        3, max_channel, height_start, height_end, width_start, width_end, x_ctxt_vec);
    */

    GoldenConv2d(image_3d, filter_3d, 1, 1);


    Encrypt_MCSR_P(image_3d, numSlots, depth, cryptoContext, height_start, height_end, width_start, width_end, keyPair, x_ctxt_vec);
    types::vector2d<Ciphertext<DCRTPoly>> x_ctxt_2d;
    x_ctxt_2d.resize(height_end - height_start + 1);

    for(int i = 0; i < height_end - height_start + 1; i++){
        // height x stored rows of multiple channels
        x_ctxt_2d[i].push_back(x_ctxt_vec[i]);
    }

    std::cout << "finished encryption" << std::endl;

    std::cout << "create the model:" << std::endl;
    // create the model
    Network model;
    //model.add_layer(std::make_shared<Conv2d>(CONV_2D, "conv1", filter_3d, bias, 1, 0, numSlots));
    model.add_layer(std::make_shared<Conv2d_P>(CONV_2D, "conv1", filter_3d, bias, 1, 1, numSlots));
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