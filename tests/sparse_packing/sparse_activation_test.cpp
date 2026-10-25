#include "../../src/cnn/sparse_activation.hpp"
#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/encryption/setparam.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        constexpr uint32_t ring = 16384, slots = 1024;
        CCParams<CryptoContextCKKSRNS> p;
        SetParamCKKS(ring, p, slots);
        p.SetMultiplicativeDepth(28);
        CRYPTOCONTEXT = GenCryptoContext(p);
        for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, SCHEMESWITCH}) CRYPTOCONTEXT->Enable(f);
        KEYPAIR = CRYPTOCONTEXT->KeyGen();
        CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
        CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);
        SchSwchParams switchParams;
        SetParam_TFHE(ring, switchParams, slots);
        std::cout << "Setting up scheme switch: ring=" << ring << " slots=" << slots << std::endl;
        auto fheKey = CRYPTOCONTEXT->EvalSchemeSwitchingSetup(switchParams);
        CCLWE = CRYPTOCONTEXT->GetBinCCForSchemeSwitch();
        CCLWE->BTKeyGen(fheKey);
        CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR, fheKey);
        const auto beta = CCLWE->GetBeta().ConvertToInt();
        CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1u << 25) / (2 * beta), 1.0);
        RESCALE_REQUIRED = false;
        sparse_activation relu(RELU_SS_ACTIVATION, "sparse_relu", fheKey, ring, slots);

        // Two blocks: channel 1 reaches slot 1023; the final block has one channel.
        double3d pts(3, types::double2d(2, std::vector<double>(512, -0.25)));
        ENCRYPTED_HEIGHT_START=ENCRYPTED_HEIGHT_END=1;
        ENCRYPTED_WIDTH_START=510; ENCRYPTED_WIDTH_END=511;
        const std::vector<double> samples = {-0.8, 0.6, 0.0, 1.2, -0.4, 0.2};
        types::vector2d<Ciphertext<DCRTPoly>> input(1), output;
        for (int block=0;block<2;++block) {
            std::vector<double> values(slots, 0);
            for (int c=2*block;c<std::min(3,2*block+2);++c) for(int w=510;w<=511;++w) {
                values[(c%2)*512+w] = samples[c*2+w-510];
                pts[c][1][w]=0;
            }
            input[0].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
                CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
        }
        double3d result;
        relu.forward(input, pts, output, result);
        double maxError=0;
        for (int block=0;block<2;++block) {
            if(output[0][block]->GetSlots()!=slots) throw std::runtime_error("ReLU changed slot count");
            Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, output[0][block], &decoded);
            decoded->SetLength(slots); auto values=decoded->GetRealPackedValue();
            for(uint32_t s=0;s<slots;++s) {
                int c=block*2+s/512, w=s%512;
                double expected = (c<3 && w>=510) ? std::max(0.0,samples[c*2+w-510]) : 0.0;
                if(!std::isfinite(values[s])) throw std::runtime_error("non-finite ReLU output");
                maxError=std::max(maxError,std::abs(values[s]-expected));
            }
        }
        for(const auto& c:result) for(const auto& row:c) for(double value:row)
            if(value!=0) throw std::runtime_error("public ReLU mismatch");
        if(maxError>1e-3) throw std::runtime_error("sparse ReLU mismatch");
        std::cout << "sparse_relu max_error=" << maxError << std::endl;

        // Convolution -> ReLU, using actual sparse convolution output at nonzero levels.
        ENCRYPTED_HEIGHT_START=ENCRYPTED_HEIGHT_END=1;
        ENCRYPTED_WIDTH_START=1; ENCRYPTED_WIDTH_END=2;
        pts.assign(3, types::double2d(3, std::vector<double>(5, 0.25)));
        std::vector<double> values(slots, 0);
        for(int c=0;c<3;++c) for(int w=1;w<=2;++w) {
            values[c*5+w]=samples[c*2+w-1]; pts[c][1][w]=0;
        }
        input={{CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
            CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots))}};
        types::double4d weights(4, double3d(3, types::double2d(1,std::vector<double>(1,0))));
        for(int c=0;c<4;++c) weights[c][c%3][0][0]=1;
        std::vector<double> gamma(4,1), zero(4,0), var(4,1);
        SparseConv2dBN_P conv(CONV_2D,"conv_relu_test",ring,slots,weights,1,0,
            gamma,zero,zero,var,1e-5,zero,CONV_2D_BN);
        types::vector2d<Ciphertext<DCRTPoly>> convCts; double3d convPts;
        conv.EnableFastPath(5);
        conv.forward(input,pts,convCts,convPts);
        relu.forward(convCts,convPts,output,result);
        Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,output[0][0],&decoded);
        decoded->SetLength(slots); values=decoded->GetRealPackedValue();
        double chainError=0;
        for(int c=0;c<4;++c) for(int h=0;h<3;++h) for(int w=0;w<5;++w) {
            double expected = (h==1 && w>=1 && w<=2) ? samples[(c%3)*2+w-1] : 0.25;
            expected=std::max(0.0,expected/std::sqrt(1+1e-5));
            double actual = (h==1 && w>=1 && w<=2) ? values[c*5+w] : result[c][h][w];
            if(!std::isfinite(actual)) throw std::runtime_error("non-finite conv/ReLU output");
            chainError=std::max(chainError,std::abs(actual-expected));
        }
        if(chainError>1e-3) throw std::runtime_error("conv/ReLU mismatch");
        std::cout << "conv_relu max_error=" << chainError << std::endl;
        // Verify repeated encrypted-mask products at nonzero levels, since the
        // old plaintext-mask depth budget cannot be reused for this path.
        for (int stage=2; stage<=3; ++stage) {
            auto previousCts=output;
            auto previousPts=result;
            types::double4d identity(4,double3d(4,types::double2d(1,std::vector<double>(1,0))));
            for(int c=0;c<4;++c) identity[c][c][0][0]=1;
            SparseConv2dBN_P next(CONV_2D,"chain",ring,slots,identity,1,0,
                gamma,zero,zero,var,1e-5,zero,CONV_2D_BN);
            next.EnableFastPath(5);
            next.forward(previousCts,previousPts,convCts,convPts);
            relu.forward(convCts,convPts,output,result);
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,output[0][0],&decoded);
            decoded->SetLength(slots); values=decoded->GetRealPackedValue();
            double error=0;
            for(int c=0;c<4;++c) for(int h=0;h<3;++h) for(int w=0;w<5;++w) {
                double expected=(h==1&&w>=1&&w<=2)?samples[(c%3)*2+w-1]:0.25;
                expected=std::max(0.0,expected/std::pow(std::sqrt(1+1e-5),stage));
                double actual=(h==1&&w>=1&&w<=2)?values[c*5+w]:result[c][h][w];
                if(!std::isfinite(actual)) throw std::runtime_error("non-finite chained ReLU output");
                error=std::max(error,std::abs(actual-expected));
            }
            std::cout<<"repeated_conv_relu stage="<<stage<<" max_error="<<error
                     <<" level="<<output[0][0]->GetLevel()<<std::endl;
            if(error>1e-3) throw std::runtime_error("repeated encrypted ReLU mismatch");
        }

    } catch(const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
