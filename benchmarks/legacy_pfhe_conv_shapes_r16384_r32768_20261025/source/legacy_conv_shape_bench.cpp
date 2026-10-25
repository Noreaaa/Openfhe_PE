// Shape microbenchmark of the unmodified legacy PFHE Conv2dBN_P implementation.
#include "../../src/cnn/conv2d.hpp"
#include "../../src/init/encrypted_region.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/experiment_phase.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <omp.h>
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){try{
 std::string root=argc>1?argv[1]:".";int target=argc>2?std::stoi(argv[2]):3;uint32_t ring=argc>3?std::stoul(argv[3]):16384,slots=argc>4?std::stoul(argv[4]):ring/2;int depth=argc>5?std::stoi(argv[5]):30,area=argc>6?std::stoi(argv[6]):10;std::string profile=argc>7?argv[7]:"legacy_shape.csv";
 if(target<1||target>18||area<1||area>224)throw std::invalid_argument("invalid layer/area");omp_set_dynamic(0);omp_set_num_threads(96);
 struct Spec{std::string w,bn;int s,p;};std::vector<Spec> specs={{"conv1","bn1",2,3},{"maxpool_0","maxpool_1",2,1}};
 for(int stage=1;stage<=4;++stage)for(int block=0;block<2;++block)for(int conv=1;conv<=2;++conv){std::string name="layer"+std::to_string(stage)+"_"+std::to_string(block);specs.push_back({name+"_conv"+std::to_string(conv),name+"_bn"+std::to_string(conv),(stage>1&&block==0&&conv==1)?2:1,1});}
 const std::string wr=root+"/python_model/saved_models/ResNet-18/parameters/";WriteExperimentPhase(profile,"reference_setup",target);
 auto input=load_bin_image_double(root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin");int start=(224-area)/2;EncryptedRegion region{start,start+area-1,start,start+area-1};std::vector<double>g,b,m,v;types::double4d f;
 for(int i=0;i<target;++i){auto spec=specs[i];f=LoadConv2dWeight(wr+spec.w+"_weight.npy");LoadConv2dBias(wr+spec.bn+"_weight.npy",g);LoadConv2dBias(wr+spec.bn+"_bias.npy",b);LoadConv2dBias(wr+spec.bn+"_running_mean.npy",m);LoadConv2dBias(wr+spec.bn+"_running_var.npy",v);if(i==target-1)break;region=MapEncryptedConvRegion(region,input[0].size(),input[0][0].size(),f[0][0].size(),f[0][0][0].size(),spec.s,spec.p);input=GoldenConv2d(input,f,std::vector<double>(f.size(),0),spec.s,spec.p);GoldenBN(input,g,b,m,v,1e-5);for(auto&c:input)for(auto&row:c)for(auto&x:row)x=std::max(0.,x);}
 int ci=input.size(),height=input[0].size(),width=input[0][0].size(),co=f.size();auto spec=specs[target-1];
 std::cout<<"SHAPE_CONFIG layer="<<target<<" weight="<<spec.w<<" input="<<ci<<'x'<<height<<'x'<<width<<" output_channels="<<co<<" kernel="<<f[0][0].size()<<" stride="<<spec.s<<" padding="<<spec.p<<" ring="<<ring<<" slots="<<slots<<" depth="<<depth<<" input_level=0 input_degree=1 original_area="<<area<<" input_region="<<region.hs<<':'<<region.he<<':'<<region.ws<<':'<<region.we<<std::endl;
 WriteExperimentPhase(profile,"key_setup",target);CCParams<CryptoContextCKKSRNS> p;SetParamCKKS(ring,p,slots);p.SetMultiplicativeDepth(depth);p.SetScalingModSize(30);p.SetFirstModSize(40);CRYPTOCONTEXT=GenCryptoContext(p);for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE})CRYPTOCONTEXT->Enable(feature);KEYPAIR=CRYPTOCONTEXT->KeyGen();CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);RESCALE_REQUIRED=false;USE_COMPACT=false;CONV_BIAS=false;
 ENCRYPTED_HEIGHT_START=region.hs;ENCRYPTED_HEIGHT_END=region.he;ENCRYPTED_WIDTH_START=region.ws;ENCRYPTED_WIDTH_END=region.we;
 auto pts=input;int per=slots/width,blocks=(ci+per-1)/per;types::vector2d<Ciphertext<DCRTPoly>> x(region.he-region.hs+1),y;
 WriteExperimentPhase(profile,"input_encryption",target);
 for(int h=region.hs;h<=region.he;++h)for(int block=0;block<blocks;++block){std::vector<double> values(slots,0);for(int c=block*per;c<std::min(ci,(block+1)*per);++c)for(int w=region.ws;w<=region.we;++w){values[(c%per)*width+w]=pts[c][h][w];pts[c][h][w]=0;}x[h-region.hs].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));}
 std::vector<double>bias(co,0);Conv2dBN_P layer(CONV_2D,"legacy_conv_shape",f,spec.s,spec.p,slots,g,b,m,v,1e-5,bias,CONV_2D_BN);
 WriteExperimentPhase(profile,"server_conv",target);auto t=Clock::now();types::double3d out;layer.forward(x,pts,y,out);double convms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();WriteExperimentPhase(profile,"verification",target);
 const auto outputRegion=MapEncryptedConvRegion(region,height,width,f[0][0].size(),f[0][0][0].size(),spec.s,spec.p);if(ENCRYPTED_HEIGHT_START!=outputRegion.hs||ENCRYPTED_HEIGHT_END!=outputRegion.he||ENCRYPTED_WIDTH_START!=outputRegion.ws||ENCRYPTED_WIDTH_END!=outputRegion.we)throw std::runtime_error("legacy encrypted-region propagation mismatch");
 int oh=out[0].size(),ow=out[0][0].size(),op=slots/ow;size_t ciphertexts=0;
 for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h)for(size_t block=0;block<y[h-ENCRYPTED_HEIGHT_START].size();++block){++ciphertexts;Plaintext plain;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,y[h-ENCRYPTED_HEIGHT_START][block],&plain);auto values=plain->GetRealPackedValue();for(int c=block*op;c<std::min(co,int((block+1)*op));++c)for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w)out[c][h][w]=values[(c%op)*ow+w];}
 auto golden=GoldenConv2d(input,f,bias,spec.s,spec.p);GoldenBN(golden,g,b,m,v,1e-5);double error=0;long count=0;long double sq=0;
 for(int c=0;c<co;++c)for(int h=0;h<oh;++h)for(int w=0;w<ow;++w){double e=out[c][h][w]-golden[c][h][w];if(!std::isfinite(e))throw std::runtime_error("nonfinite output");error=std::max(error,std::abs(e));sq+=e*e;++count;}
 auto ct=y[0][0];rusage usage{};getrusage(RUSAGE_SELF,&usage);double rmse=std::sqrt(sq/count);
 std::ofstream csv(profile);csv<<"layer,ring,slots,depth,input_channels,input_height,input_width,output_channels,output_height,output_width,kernel,stride,encrypted_height,encrypted_width,conv_ms,ciphertexts,output_level,towers,max_abs,rmse,whole_peak_gib\n"<<std::setprecision(12)<<target<<','<<ring<<','<<slots<<','<<depth<<','<<ci<<','<<height<<','<<width<<','<<co<<','<<oh<<','<<ow<<','<<f[0][0].size()<<','<<spec.s<<','<<outputRegion.he-outputRegion.hs+1<<','<<outputRegion.we-outputRegion.ws+1<<','<<convms<<','<<ciphertexts<<','<<ct->GetLevel()<<','<<ct->GetElements()[0].GetNumOfElements()<<','<<error<<','<<rmse<<','<<usage.ru_maxrss/1048576.<<'\n';csv.flush();
 std::cout<<"SHAPE_RESULT layer="<<target<<" ring="<<ring<<" conv_s="<<convms/1000<<" peak_gib="<<usage.ru_maxrss/1048576.<<" max_abs="<<error<<" rmse="<<rmse<<std::endl;
 if(error>.02||rmse>.001)throw std::runtime_error("legacy convolution numeric check failed");WriteExperimentPhase(profile,"finished",target);return 0;
}catch(const std::exception&e){std::cerr<<"SHAPE_ERROR "<<e.what()<<std::endl;return 1;}}
