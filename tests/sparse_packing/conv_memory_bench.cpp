#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/cnn/conv2d.hpp"
#include <chrono>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>
#include <omp.h>
#include <malloc.h>

static double rssMiB() {
    std::ifstream stream("/proc/self/statm"); long total=0,resident=0;stream>>total>>resident;
    return resident*static_cast<double>(sysconf(_SC_PAGESIZE))/(1024*1024);
}
static double peakMiB() {rusage r{};getrusage(RUSAGE_SELF,&r);return r.ru_maxrss/1024.;}
static void allocatorStats(const char* phase) {
    const auto m=mallinfo2();
    std::cout<<"ALLOCATOR phase="<<phase<<" rss_mib="<<rssMiB()
             <<" heap_inuse_mib="<<m.uordblks/1048576.0<<" heap_free_mib="<<m.fordblks/1048576.0
             <<" mmap_live_mib="<<m.hblkhd/1048576.0<<" arena_mib="<<m.arena/1048576.0<<std::endl;
}
int main(int argc,char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"optimized";
        if(mode!="baseline" && mode!="optimized") throw std::invalid_argument("mode must be baseline or optimized");
        const uint32_t budget=argc>2?std::stoul(argv[2]):2048;
        const bool allocatorProbe=argc>3 && std::string(argv[3])=="allocator";
        constexpr uint32_t ring=32768,slots=2048;
        constexpr int ci=8,co=32,width=32,kernel=3,stride=2,padding=1;
        CCParams<CryptoContextCKKSRNS> p;
        p.SetSecurityLevel(HEStd_NotSet);p.SetRingDim(ring);p.SetBatchSize(slots);
        p.SetSecretKeyDist(UNIFORM_TERNARY);p.SetScalingTechnique(FLEXIBLEAUTO);
        p.SetScalingModSize(30);p.SetFirstModSize(40);p.SetMultiplicativeDepth(40);p.SetNumLargeDigits(6);
        CRYPTOCONTEXT=GenCryptoContext(p);
        for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE})CRYPTOCONTEXT->Enable(f);
        KEYPAIR=CRYPTOCONTEXT->KeyGen();
        double3d original(ci,types::double2d(width,std::vector<double>(width)));
        for(int c=0;c<ci;++c)for(int h=0;h<width;++h)for(int w=0;w<width;++w)
            original[c][h][w]=0.1*std::sin(c*0.3+h*0.13+w*0.17);
        auto pts=original;
        ENCRYPTED_HEIGHT_START=12;ENCRYPTED_HEIGHT_END=19;
        ENCRYPTED_WIDTH_START=12;ENCRYPTED_WIDTH_END=19;
        USE_COMPACT=false;RESCALE_REQUIRED=false;
        types::vector2d<Ciphertext<DCRTPoly>> x(8),y;
        for(int h=12;h<=19;++h) {
            std::vector<double> values(slots,0);
            for(int c=0;c<ci;++c)for(int w=12;w<=19;++w) {values[c*width+w]=pts[c][h][w];pts[c][h][w]=0;}
            x[h-12].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
                CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
        }
        types::double4d weights(co,double3d(ci,types::double2d(kernel,std::vector<double>(kernel))));
        for(int o=0;o<co;++o)for(int c=0;c<ci;++c)for(int h=0;h<kernel;++h)for(int w=0;w<kernel;++w)
            weights[o][c][h][w]=0.01*std::cos(o*0.1+c+h-w);
        std::vector<double> gamma(co,1.1),beta(co,0.01),mean(co,0.02),var(co,0.9),bias(co,0);
        SparseConv2dBN_P layer(CONV_2D,"memory_bench",ring,slots,weights,stride,padding,gamma,beta,mean,var,1e-5,bias,CONV_2D_BN);
        auto start=std::chrono::steady_clock::now();
        layer.EnableFastPath(width,true,mode=="optimized",mode=="optimized"?budget:0);
        const double keySeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const auto& keys=CRYPTOCONTEXT->GetEvalAutomorphismKeyMap(KEYPAIR.secretKey->GetKeyTag());
        uint64_t keyBytes=0;
        for(const auto& entry:keys)for(const auto* v:{&entry.second->GetAVector(),&entry.second->GetBVector()})
            for(const auto& poly:*v)keyBytes+=static_cast<uint64_t>(poly.GetRingDimension())*poly.GetNumOfElements()*sizeof(uint64_t);
        std::cout<<"MEMORY_KEYS mode="<<mode<<" keys="<<keys.size()<<" coefficient_mib="<<keyBytes/1048576.0
                 <<" rss_mib="<<rssMiB()<<" peak_mib="<<peakMiB()<<" key_s="<<keySeconds<<" threads="<<omp_get_max_threads()<<std::endl;
        if(allocatorProbe)allocatorStats("after_keys");
        start=std::chrono::steady_clock::now();double3d out;
        layer.forward(x,pts,y,out);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const double forwardPeak=peakMiB();
        if(allocatorProbe)allocatorStats("after_forward_return");
        const int ow=out[0][0].size(),per=slots/ow;
        for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h)
            for(size_t block=0;block<y[h-ENCRYPTED_HEIGHT_START].size();++block) {
                Plaintext plain;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,y[h-ENCRYPTED_HEIGHT_START][block],&plain);
                plain->SetLength(slots);const auto values=plain->GetRealPackedValue();
                for(int c=block*per;c<std::min(co,static_cast<int>((block+1)*per));++c)
                    for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w)out[c][h][w]=values[(c%per)*ow+w];
            }
        auto expected=GoldenConv2d(original,weights,bias,stride,padding);GoldenBN(expected,gamma,beta,mean,var,1e-5);
        double error=0;
        for(int c=0;c<co;++c)for(int h=0;h<ow;++h)for(int w=0;w<ow;++w){
            if(!std::isfinite(out[c][h][w]))throw std::runtime_error("nonfinite output");
            error=std::max(error,std::abs(out[c][h][w]-expected[c][h][w]));
        }
        std::cout<<"MEMORY_RESULT mode="<<mode<<" conv_s="<<seconds<<" forward_peak_mib="<<forwardPeak
                 <<" process_peak_mib="<<peakMiB()<<" max_abs="<<error<<std::endl;
        if(error>1e-3)throw std::runtime_error("plaintext reference mismatch");
        if(allocatorProbe) {
            allocatorStats("before_diagnostic_trim");
            // Diagnostic only, after timing and correctness checks; never used by production.
            const int released=malloc_trim(0);
            std::cout<<"DIAGNOSTIC_TRIM released="<<released<<std::endl;
            allocatorStats("after_diagnostic_trim");
        }
    } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
