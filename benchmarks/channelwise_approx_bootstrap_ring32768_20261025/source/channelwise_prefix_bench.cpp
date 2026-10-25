// Channel-wise baseline derived from Conv2d_C: one ciphertext per channel,
// row-major spatial indices retained across strided convolutions.
#include "../../src/cnn/conv2d.hpp"
#include "../../src/cnn/encrypted_relu.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/utils/experiment_phase.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sys/resource.h>
#include <omp.h>
using namespace lbcrypto;
using CT=Ciphertext<DCRTPoly>;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
struct Layout {int h,w,pitch,step;int pos(int h0,int w0)const{return h0*step*pitch+w0*step;}};
static std::vector<size_t> active(const Layout& l){std::vector<size_t> a;for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w)a.push_back(l.pos(h,w));return a;}
static void bnload(const std::string& root,const std::string& name,std::vector<double>& g,std::vector<double>& b,std::vector<double>& m,std::vector<double>& v){LoadConv2dBias(root+name+"_weight.npy",g);LoadConv2dBias(root+name+"_bias.npy",b);LoadConv2dBias(root+name+"_running_mean.npy",m);LoadConv2dBias(root+name+"_running_var.npy",v);}
static types::double3d plainconv(const types::double3d& x,types::double4d& f,int s,int p,std::vector<double>&g,std::vector<double>&b,std::vector<double>&m,std::vector<double>&v){auto y=GoldenConv2d(x,f,std::vector<double>(f.size(),0),s,p);GoldenBN(y,g,b,m,v,1e-5);return y;}
static double maxabs(const types::double3d& x){double r=0;for(auto&c:x)for(auto&row:c)for(auto v:row)r=std::max(r,std::abs(v));return r;}
static double polynomial(double x){double z=x*x;return .0229645+.5*x+z*(1.4340124+z*(-2.089185+z*1.1551725));}
static std::vector<CT> encrypt(const types::double3d& x,const Layout& l,uint32_t slots){std::vector<CT> out(x.size());
#pragma omp parallel for
for(int c=0;c<(int)x.size();++c){std::vector<double> v(slots);for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w)v[l.pos(h,w)]=x[c][h][w];out[c]=CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(v,1,0,nullptr,slots));}return out;}
// Reuse rotations across output channels; each plaintext incorporates padding,
// sampling anchors, trained kernel and folded BN. No ciphertext channel mixing.
static std::vector<CT> convolution(const std::vector<CT>&x,Layout& l,types::double4d&f,int stride,int padding,std::vector<double>&g,std::vector<double>&b,std::vector<double>&m,std::vector<double>&v,uint32_t slots,double divisor){
 int kh=f[0][0].size(),kw=f[0][0][0].size(),ci=x.size(),co=f.size();
 Layout out{(l.h+2*padding-kh)/stride+1,(l.w+2*padding-kw)/stride+1,l.pitch,l.step*stride};
 std::vector<std::vector<CT>> rotated(ci,std::vector<CT>(kh*kw));
#pragma omp parallel for
 for(int c=0;c<ci;++c)for(int h=0;h<kh;++h)for(int w=0;w<kw;++w){int off=(h-padding)*l.step*l.pitch+(w-padding)*l.step;rotated[c][h*kw+w]=off?CRYPTOCONTEXT->EvalRotate(x[c],off):x[c];}
 std::vector<std::vector<int>> positions(kh*kw);
 for(int h=0;h<kh;++h)for(int w=0;w<kw;++w)for(int oh=0;oh<out.h;++oh)for(int ow=0;ow<out.w;++ow){int ih=oh*stride+h-padding,iw=ow*stride+w-padding;if(ih>=0&&ih<l.h&&iw>=0&&iw<l.w)positions[h*kw+w].push_back(out.pos(oh,ow));}
 std::vector<CT> y(co);
#pragma omp parallel for
 for(int oc=0;oc<co;++oc){CT sum;double scale=g[oc]/std::sqrt(v[oc]+1e-5)/divisor;
   for(int ic=0;ic<ci;++ic)for(int h=0;h<kh;++h)for(int w=0;w<kw;++w){std::vector<double> vals(slots,0);double k=f[oc][ic][h][w]*scale;for(int pos:positions[h*kw+w])vals[pos]=k;
     auto pt=CRYPTOCONTEXT->MakeCKKSPackedPlaintext(vals,1,x[ic]->GetLevel(),nullptr,slots);auto term=CRYPTOCONTEXT->EvalMult(rotated[ic][h*kw+w],pt);if(sum)CRYPTOCONTEXT->EvalAddInPlace(sum,term);else sum=term;
   }
   std::vector<double> bias(slots,0);double biasv=(b[oc]-g[oc]*m[oc]/std::sqrt(v[oc]+1e-5))/divisor;for(auto pos:active(out))bias[pos]=biasv;
   y[oc]=CRYPTOCONTEXT->EvalAdd(sum,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(bias,sum->GetNoiseScaleDeg(),sum->GetLevel(),nullptr,slots));
 }
 l=out;return y;
}
struct Errors{double max=0,rmse=0;};
static Errors verify(const std::vector<CT>&cts,const Layout&l,const types::double3d&ref){long n=0;long double sq=0;Errors e;
 for(size_t c=0;c<cts.size();++c){Plaintext p;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,cts[c],&p);auto vals=p->GetRealPackedValue();for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w){double actual=vals.at(l.pos(h,w)),err=std::abs(actual-ref[c][h][w]);if(!std::isfinite(actual))throw std::runtime_error("nonfinite decrypted output");e.max=std::max(e.max,err);sq+=err*err;++n;}}
 e.rmse=std::sqrt(sq/n);return e;}
static void rotations(const std::vector<int>& pitches,const std::vector<int>&steps){std::set<int32_t> keys;for(int pitch:pitches)for(int step:steps)for(int h=-1;h<=1;++h)for(int w=-1;w<=1;++w)if(int o=h*step*pitch+w*step)keys.insert(o);CRYPTOCONTEXT->EvalRotateKeyGen(KEYPAIR.secretKey,std::vector<int32_t>(keys.begin(),keys.end()));}
int main(int argc,char**argv){try{
 std::string root=argc>1?argv[1]:".",mode=argc>2?argv[2]:"scheme_switch",profile=argc>3?argv[3]:"channelwise.csv";
 bool smoke=mode=="smoke";bool ss=mode=="scheme_switch";bool boot=mode=="approximation_bootstrap"||mode=="bootstrap_probe";if(!smoke&&!ss&&!boot&&mode!="approximation")throw std::invalid_argument("unknown benchmark mode");
 int last=argc>4?std::stoi(argv[4]):8;int depth=argc>5?std::stoi(argv[5]):(boot?0:(ss?40:36));uint32_t ring=smoke?2048:32768,slots=ring/2;
 int scaleBits=argc>6?std::stoi(argv[6]):30,firstBits=argc>7?std::stoi(argv[7]):40;std::vector<uint32_t> budget{3,3};int reserve=boot?FHECKKSRNS::GetBootstrapDepth(budget,UNIFORM_TERNARY):0;if(boot&&depth==0)depth=reserve+10;
 omp_set_dynamic(0);omp_set_num_threads(96);
 WriteExperimentPhase(profile,"key_setup");CCParams<CryptoContextCKKSRNS> params;SetParamCKKS(ring,params,slots);params.SetMultiplicativeDepth(smoke?16:depth);params.SetScalingModSize(scaleBits);params.SetFirstModSize(firstBits);
 CRYPTOCONTEXT=GenCryptoContext(params);for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,SCHEMESWITCH})CRYPTOCONTEXT->Enable(feature);if(boot){CRYPTOCONTEXT->Enable(FHE);CRYPTOCONTEXT->EvalBootstrapSetup(budget,{0,0},slots,0);}KEYPAIR=CRYPTOCONTEXT->KeyGen();CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);if(boot)CRYPTOCONTEXT->EvalBootstrapKeyGen(KEYPAIR.secretKey,slots);
 if(ss){SchSwchParams sw;SetParam_TFHE(ring,sw,slots);auto sk=CRYPTOCONTEXT->EvalSchemeSwitchingSetup(sw);CCLWE=CRYPTOCONTEXT->GetBinCCForSchemeSwitch();CCLWE->BTKeyGen(sk);CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR,sk);auto beta=CCLWE->GetBeta().ConvertToInt();CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1<<25)/(2*beta),1.0);}
 rotations({smoke?8:112},{1,2,4});
 std::cout<<"CHANNELWISE_CONFIG mode="<<mode<<" ring="<<ring<<" slots="<<slots<<" depth="<<depth<<" scale_bits="<<scaleBits<<" first_bits="<<firstBits<<" digits=6 threads=96 bootstrap="<<(boot?"after_every_activation":"none")<<" bootstrap_reserve="<<reserve<<" last_layer="<<last<<std::endl;
 if(mode=="bootstrap_probe"){
 Layout l{56,56,112,2};types::double3d reference(1,types::double2d(56,std::vector<double>(56)));for(int h=0;h<56;++h)for(int w=0;w<56;++w)reference[0][h][w]=.6*std::sin(h*.13+w*.07);auto cts=encrypt(reference,l,slots);
 for(int pass=0;pass<2;++pass){auto t=Clock::now();auto y=CRYPTOCONTEXT->EvalPoly(cts[0],{0.,.5,1.4340124,0.,-2.089185,0.,1.1551725});std::vector<double> bias(slots,0);for(auto pos:active(l))bias[pos]=.0229645;y=CRYPTOCONTEXT->EvalAdd(y,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(bias,y->GetNoiseScaleDeg(),y->GetLevel(),nullptr,slots));for(auto&row:reference[0])for(auto&val:row)val=polynomial(val);
 auto normalized=CRYPTOCONTEXT->EvalMult(y,1.0);auto towers=normalized->GetElements()[0].GetNumOfElements();if(normalized->GetNoiseScaleDeg()>1)CRYPTOCONTEXT->GetScheme()->ModReduceInternalInPlace(normalized,normalized->GetNoiseScaleDeg()-1);towers=normalized->GetElements()[0].GetNumOfElements();if(towers>2)CRYPTOCONTEXT->GetScheme()->LevelReduceInternalInPlace(normalized,towers-2);if(normalized->GetElements()[0].GetNumOfElements()!=2)throw std::runtime_error("bootstrap input was not depleted to two towers");auto depletedError=verify(std::vector<CT>{normalized},l,reference);std::cout<<"PROBE_DEPLETED level="<<normalized->GetLevel()<<" towers="<<normalized->GetElements()[0].GetNumOfElements()<<" degree="<<normalized->GetNoiseScaleDeg()<<" rmse="<<depletedError.rmse<<std::endl;auto refreshed=CRYPTOCONTEXT->EvalBootstrap(normalized,1,0);if(refreshed->GetElements()[0].GetNumOfElements()<=2)throw std::runtime_error("bootstrap did not restore towers");cts[0]=CRYPTOCONTEXT->EvalMult(refreshed,1.0);auto e=verify(cts,l,reference);std::cout<<"BOOTSTRAP_PROBE pass="<<pass<<" max_abs="<<e.max<<" rmse="<<e.rmse<<" level="<<cts[0]->GetLevel()<<" towers="<<cts[0]->GetElements()[0].GetNumOfElements()<<" elapsed_ms="<<elapsed(t)<<std::endl;if(e.rmse>.005||cts[0]->GetElements()[0].GetNumOfElements()<7)throw std::runtime_error("bootstrap probe accuracy or remaining-budget check failed");}
 WriteExperimentPhase(profile,"finished");return 0;}
 if(smoke){Layout l{8,8,8,1};types::double3d ref(2,types::double2d(8,std::vector<double>(8)));for(int c=0;c<2;++c)for(int h=0;h<8;++h)for(int w=0;w<8;++w)ref[c][h][w]=std::sin(c+h*.2+w*.3);auto cts=encrypt(ref,l,slots);
 for(int pass=0;pass<2;++pass){int ci=ref.size(),co=pass?2:3,stride=pass?1:2;types::double4d f(co,types::double3d(ci,types::double2d(3,std::vector<double>(3))));for(int o=0;o<co;++o)for(int i=0;i<ci;++i)for(int h=0;h<3;++h)for(int w=0;w<3;++w)f[o][i][h][w]=.01*(o+1)*(i+1)*(h+w+1);std::vector<double> g(co,.7),b(co,.1),m(co,.03),v(co,.9);ref=plainconv(ref,f,stride,1,g,b,m,v);cts=convolution(cts,l,f,stride,1,g,b,m,v,slots,1);auto e=verify(cts,l,ref);std::cout<<"SMOKE_CONV pass="<<pass<<" max_abs="<<e.max<<std::endl;if(e.max>.001)throw std::runtime_error("channel-wise padded convolution validation failed");}
 std::vector<CT> y(cts.size());double R=std::max(1.,maxabs(ref)*1.1);for(size_t c=0;c<cts.size();++c){auto z=CRYPTOCONTEXT->EvalMult(cts[c],1/R);y[c]=CRYPTOCONTEXT->EvalPoly(z,{0.,.5*R,1.4340124*R,0.,-2.089185*R,0.,1.1551725*R});std::vector<double> constant(slots);for(auto pos:active(l))constant[pos]=.0229645*R;y[c]=CRYPTOCONTEXT->EvalAdd(y[c],CRYPTOCONTEXT->MakeCKKSPackedPlaintext(constant,y[c]->GetNoiseScaleDeg(),y[c]->GetLevel(),nullptr,slots));for(auto&row:ref[c])for(auto&val:row)val=R*polynomial(val/R);}auto e=verify(y,l,ref);if(e.max>.001)throw std::runtime_error("approximation arithmetic validation failed");std::cout<<"SMOKE_PASSED approximation_max_abs="<<e.max<<std::endl;WriteExperimentPhase(profile,"finished");return 0;}
 const std::string wr=root+"/python_model/saved_models/ResNet-18/parameters/";
 WriteExperimentPhase(profile,"client_first",1);auto start=Clock::now();auto ref=load_bin_image_double(root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin");std::vector<double>g,b,m,v;auto first=LoadConv2dWeight(wr+"conv1_weight.npy");bnload(wr,"bn1",g,b,m,v);ref=plainconv(ref,first,2,3,g,b,m,v);for(auto&c:ref)for(auto&row:c)for(auto&val:row)val=std::max(0.,val);std::cout<<"CLIENT_FIRST ms="<<elapsed(start)<<std::endl;
 Layout layout{112,112,112,1};WriteExperimentPhase(profile,"input_encryption");start=Clock::now();auto cts=encrypt(ref,layout,slots);std::cout<<"ENCRYPTION ms="<<elapsed(start)<<" ciphertexts="<<cts.size()<<std::endl;
 struct Spec{const char*w,*bn;int stride;};std::vector<Spec> specs={{"maxpool_0","maxpool_1",2},{"layer1_0_conv1","layer1_0_bn1",1},{"layer1_0_conv2","layer1_0_bn2",1},{"layer1_1_conv1","layer1_1_bn1",1},{"layer1_1_conv2","layer1_1_bn2",1},{"layer2_0_conv1","layer2_0_bn1",2},{"layer2_0_conv2","layer2_0_bn2",1}};
 std::ofstream csv(profile);csv<<"layer,conv_ms,activation_ms,cumulative_ms,channels,height,width,ciphertexts,level,remaining_towers,normalization_bound,conv_max_abs,conv_rmse,activation_max_abs,activation_rmse,vs_relu_max_abs,vs_relu_rmse,bootstrap_ms,pre_bootstrap_level,post_bootstrap_level,bootstrap_scale,bootstrap_max_abs,bootstrap_rmse\n";double cumulative=0;
 for(int layer=2;layer<=last;++layer){auto spec=specs.at(layer-2);WriteExperimentPhase(profile,"reference_setup",layer);auto f=LoadConv2dWeight(wr+spec.w+"_weight.npy");bnload(wr,spec.bn,g,b,m,v);auto pre=plainconv(ref,f,spec.stride,1,g,b,m,v);double R=ss?1:std::max(1.,1.1*maxabs(pre));auto expectedconv=pre;for(auto&c:expectedconv)for(auto&row:c)for(auto&val:row)val/=R;
 WriteExperimentPhase(profile,"server_conv",layer);start=Clock::now();auto output=convolution(cts,layout,f,spec.stride,1,g,b,m,v,slots,R);double convms=elapsed(start);cts.clear();WriteExperimentPhase(profile,"verification",layer);auto ce=verify(output,layout,expectedconv);if(ce.max>.1||!std::isfinite(ce.max))throw std::runtime_error("convolution numeric check failed");
 auto reluref=pre;for(auto&c:reluref)for(auto&row:c)for(auto&val:row)val=std::max(0.,val);
 ref=pre;for(auto&c:ref)for(auto&row:c)for(auto&val:row)val=ss?std::max(0.,val):R*polynomial(val/R);
 WriteExperimentPhase(profile,"server_relu",layer);start=Clock::now();cts.resize(output.size());auto indices=active(layout);
 if(ss){for(size_t c=0;c<output.size();++c){auto amp=CRYPTOCONTEXT->EvalMult(output[c],1000.);auto lwe=CRYPTOCONTEXT->EvalCKKStoFHEW(amp,slots);auto mask=EncryptedReluMask(CRYPTOCONTEXT,CCLWE,lwe,indices,slots);cts[c]=CRYPTOCONTEXT->EvalMult(output[c],mask);}}
 else{
#pragma omp parallel for
 for(int c=0;c<(int)output.size();++c){auto y=CRYPTOCONTEXT->EvalPoly(output[c],{0.,.5*R,1.4340124*R,0.,-2.089185*R,0.,1.1551725*R});std::vector<double> constant(slots,0);for(auto pos:indices)constant[pos]=.0229645*R;cts[c]=CRYPTOCONTEXT->EvalAdd(y,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(constant,y->GetNoiseScaleDeg(),y->GetLevel(),nullptr,slots));}}
 double actms=elapsed(start);output.clear();double bootms=0,bootScale=1;size_t beforeBoot=cts[0]->GetLevel(),afterBoot=beforeBoot;Errors be;
 if(boot){WriteExperimentPhase(profile,"verification",layer);auto before=verify(cts,layout,ref);std::cout<<"PRE_BOOTSTRAP layer="<<layer<<" rmse="<<before.rmse<<std::endl;bootScale=std::max(1.,1.1*maxabs(ref));WriteExperimentPhase(profile,"server_bootstrap",layer);start=Clock::now();
 for(size_t c=0;c<cts.size();++c){auto normalized=CRYPTOCONTEXT->EvalMult(cts[c],1./bootScale);auto towers=normalized->GetElements()[0].GetNumOfElements();if(normalized->GetNoiseScaleDeg()>1)CRYPTOCONTEXT->GetScheme()->ModReduceInternalInPlace(normalized,normalized->GetNoiseScaleDeg()-1);towers=normalized->GetElements()[0].GetNumOfElements();if(towers>2)CRYPTOCONTEXT->GetScheme()->LevelReduceInternalInPlace(normalized,towers-2);if(normalized->GetElements()[0].GetNumOfElements()!=2)throw std::runtime_error("bootstrap input was not depleted to two towers");auto refreshed=CRYPTOCONTEXT->EvalBootstrap(normalized,1,0);if(refreshed->GetElements()[0].GetNumOfElements()<=2)throw std::runtime_error("bootstrap did not restore towers");cts[c]=CRYPTOCONTEXT->EvalMult(refreshed,bootScale);}
 bootms=elapsed(start);afterBoot=cts[0]->GetLevel();WriteExperimentPhase(profile,"verification",layer);be=verify(cts,layout,ref);}
 cumulative+=convms+actms+bootms;WriteExperimentPhase(profile,"verification",layer);auto ae=verify(cts,layout,ref),re=verify(cts,layout,reluref);if(ae.rmse>.02*std::max(1.,maxabs(ref)))throw std::runtime_error("activation arithmetic check failed");
 csv<<std::setprecision(12)<<layer<<','<<convms<<','<<actms<<','<<cumulative<<','<<cts.size()<<','<<layout.h<<','<<layout.w<<','<<cts.size()<<','<<cts[0]->GetLevel()<<','<<cts[0]->GetElements()[0].GetNumOfElements()<<','<<R<<','<<ce.max<<','<<ce.rmse<<','<<ae.max<<','<<ae.rmse<<','<<re.max<<','<<re.rmse<<','<<bootms<<','<<beforeBoot<<','<<afterBoot<<','<<bootScale<<','<<be.max<<','<<be.rmse<<'\n';csv.flush();std::cout<<"LAYER_FINISHED layer="<<layer<<" conv_ms="<<convms<<" activation_ms="<<actms<<" bootstrap_ms="<<bootms<<" max_abs="<<ae.max<<" vs_relu_rmse="<<re.rmse<<" level="<<cts[0]->GetLevel()<<std::endl;
 }
 WriteExperimentPhase(profile,"finished");return 0;
 }catch(const std::exception&e){std::cerr<<"EXPERIMENT_ERROR "<<e.what()<<std::endl;return 1;}}
