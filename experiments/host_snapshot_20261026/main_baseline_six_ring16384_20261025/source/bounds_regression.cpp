#include "conv2d.hpp"
#include <cmath>
#include <iostream>
int main(){try{
 CCParams<CryptoContextCKKSRNS> p;p.SetSecurityLevel(HEStd_NotSet);p.SetRingDim(1024);p.SetBatchSize(512);p.SetMultiplicativeDepth(12);p.SetScalingModSize(40);p.SetFirstModSize(50);p.SetScalingTechnique(FLEXIBLEAUTO);
 CRYPTOCONTEXT=GenCryptoContext(p);for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE})CRYPTOCONTEXT->Enable(f);KEYPAIR=CRYPTOCONTEXT->KeyGen();CRYPTOCONTEXT->EvalRotateKeyGen(KEYPAIR.secretKey,{1,2,16,17,18,32,33,34});
 double3d input(2,types::double2d(16,std::vector<double>(16)));std::vector<Ciphertext<DCRTPoly>> cts;
 for(int c=0;c<2;++c){std::vector<double> vals(512);for(int h=0;h<16;++h)for(int w=0;w<16;++w)vals[h*16+w]=input[c][h][w]=.1*std::sin(c+h*.2+w*.3);cts.push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(vals)));}
 update_index_maps(16,16,1,3,true);types::double4d weights(2,double3d(2,types::double2d(3,std::vector<double>(3,.02))));auto expected=input;
 for(int stage=0;stage<2;++stage){int width=16-2*stage;Conv2d_C layer(CONV_2D_C,"regression",weights,1,width,width,512);std::vector<Ciphertext<DCRTPoly>> output;layer.forward(cts,output);expected=GoldenConv2d(expected,weights,{0.,0.},1,0);double error=0;
 for(int c=0;c<2;++c){Plaintext pt;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,output[c],&pt);auto vals=pt->GetRealPackedValue();for(int h=0;h<width-2;++h)for(int w=0;w<width-2;++w)error=std::max(error,std::abs(vals.at(VALID_INDEX_MAP[h][w])-expected[c][h][w]));}
 if(error>1e-5)throw std::runtime_error("regression mismatch");cts=std::move(output);std::cout<<"BOUNDS_REGRESSION stage="<<stage+1<<" max_abs="<<error<<std::endl;}
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
