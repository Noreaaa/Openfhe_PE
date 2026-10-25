#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/cnn/conv2d.hpp"
#include "../../src/init/encrypted_region.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/encryption/setparam.hpp"
#include <chrono>
#include <sys/resource.h>
#include <omp.h>
int main(int argc,char**argv){try{
 std::string root=argc>1?argv[1]:".";int target=argc>2?std::stoi(argv[2]):2,area=argc>3?std::stoi(argv[3]):40;
 int inputLevel=argc>4?std::stoi(argv[4]):0;int inputDegree=argc>5?std::stoi(argv[5]):1;if(inputDegree!=1&&inputDegree!=2)throw std::invalid_argument("input degree must be 1 or 2");if(target<2||target>8)throw std::invalid_argument("target 2..8");
 omp_set_dynamic(0);omp_set_num_threads(96);
 const std::string wr=root+"/python_model/saved_models/ResNet-18/parameters/";
 struct Spec{const char*w,*bn;int s,p;};std::vector<Spec> specs={{"conv1","bn1",2,3},{"maxpool_0","maxpool_1",2,1},{"layer1_0_conv1","layer1_0_bn1",1,1},{"layer1_0_conv2","layer1_0_bn2",1,1},{"layer1_1_conv1","layer1_1_bn1",1,1},{"layer1_1_conv2","layer1_1_bn2",1,1},{"layer2_0_conv1","layer2_0_bn1",2,1},{"layer2_0_conv2","layer2_0_bn2",1,1}};
 auto input=load_bin_image_double(root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin");EncryptedRegion region{(224-area)/2,(224+area)/2-1,(224-area)/2,(224+area)/2-1};
 std::vector<double> g,b,m,v;types::double4d f;
 for(int index=0;index<target;++index){auto s=specs[index];f=LoadConv2dWeight(wr+s.w+"_weight.npy");LoadConv2dBias(wr+s.bn+"_weight.npy",g);LoadConv2dBias(wr+s.bn+"_bias.npy",b);LoadConv2dBias(wr+s.bn+"_running_mean.npy",m);LoadConv2dBias(wr+s.bn+"_running_var.npy",v);if(index==target-1)break;region=MapEncryptedConvRegion(region,input[0].size(),input[0][0].size(),f[0][0].size(),f[0][0][0].size(),s.s,s.p);input=GoldenConv2d(input,f,std::vector<double>(f.size(),0),s.s,s.p);GoldenBN(input,g,b,m,v,1e-5);for(auto&c:input)for(auto&row:c)for(auto&val:row)val=std::max(0.,val);}
 constexpr uint32_t ring=16384,slots=8192;CCParams<CryptoContextCKKSRNS> p;SetParamCKKS(ring,p,slots);p.SetMultiplicativeDepth(40);p.SetScalingModSize(30);p.SetFirstModSize(40);CRYPTOCONTEXT=GenCryptoContext(p);for(auto flag:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE})CRYPTOCONTEXT->Enable(flag);KEYPAIR=CRYPTOCONTEXT->KeyGen();RESCALE_REQUIRED=false;USE_COMPACT=false;
 ENCRYPTED_HEIGHT_START=region.hs;ENCRYPTED_HEIGHT_END=region.he;ENCRYPTED_WIDTH_START=region.ws;ENCRYPTED_WIDTH_END=region.we;
 auto pts=input;int width=input[0][0].size(),per=slots/width,ci=input.size(),blocks=(ci+per-1)/per;types::vector2d<Ciphertext<DCRTPoly>> x(region.he-region.hs+1),y;
 for(int h=region.hs;h<=region.he;++h)for(int block=0;block<blocks;++block){std::vector<double> values(slots,0);for(int c=block*per;c<std::min(ci,(block+1)*per);++c)for(int w=region.ws;w<=region.we;++w){values[(c%per)*width+w]=pts[c][h][w];pts[c][h][w]=0;}x[h-region.hs].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,inputLevel,nullptr,slots)));}
 if(inputDegree==2)for(auto&row:x)for(auto&ct:row)ct=CRYPTOCONTEXT->EvalMult(ct,1.0);
 auto spec=specs[target-1];SparseConv2dBN_P layer(CONV_2D,"optimization_bench",ring,slots,f,spec.s,spec.p,g,b,m,v,1e-5,std::vector<double>(f.size(),0),CONV_2D_BN);layer.EnableFastPath(width,true,false,0);
 auto start=std::chrono::steady_clock::now();types::double3d out;layer.forward(x,pts,y,out);double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 auto ct=y[0][0];size_t level=ct->GetLevel(),towers=ct->GetElements()[0].GetNumOfElements();int ow=out[0][0].size(),co=out.size(),outputPer=slots/ow;
 for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h)for(size_t block=0;block<y[h-ENCRYPTED_HEIGHT_START].size();++block){Plaintext plain;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,y[h-ENCRYPTED_HEIGHT_START][block],&plain);auto values=plain->GetRealPackedValue();for(int c=block*outputPer;c<std::min(co,int((block+1)*outputPer));++c)for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w)out[c][h][w]=values[(c%outputPer)*ow+w];}
 auto golden=GoldenConv2d(input,f,std::vector<double>(co,0),spec.s,spec.p);GoldenBN(golden,g,b,m,v,1e-5);double maxError=0;long double sq=0;long count=0;
 for(int c=0;c<co;++c)for(int h=0;h<ow;++h)for(int w=0;w<ow;++w){double e=out[c][h][w]-golden[c][h][w];if(!std::isfinite(e))throw std::runtime_error("nonfinite result");maxError=std::max(maxError,std::abs(e));sq+=e*e;++count;}
 rusage usage{};getrusage(RUSAGE_SELF,&usage);std::cout<<"OPT_BENCH layer="<<target<<" area="<<area<<" input_level="<<inputLevel<<" input_noise_degree="<<inputDegree<<" conv_s="<<sec<<" max_abs="<<maxError<<" rmse="<<std::sqrt(sq/count)<<" output_level="<<level<<" towers="<<towers<<" peak_gib="<<usage.ru_maxrss/1048576.<<std::endl;
 if(maxError>.02)throw std::runtime_error("numeric mismatch");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
