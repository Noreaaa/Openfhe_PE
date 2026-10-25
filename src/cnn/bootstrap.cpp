#include "bootstrap.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

Bootstrap_P::Bootstrap_P(PLayerType layer_type,std::string layer_name)
    : Layer(layer_type,std::move(layer_name)) {}
Bootstrap_P::~Bootstrap_P() = default;

void Bootstrap_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    if(x_cts.empty()) throw std::invalid_argument("bootstrap: empty ciphertext tensor");
    y_pts=x_pts;
    y_cts.resize(x_cts.size());
    size_t count=0;
    size_t beforeMin=std::numeric_limits<size_t>::max(),beforeMax=0,afterMin=beforeMin,afterMax=0;
    auto start=std::chrono::steady_clock::now();
    // Each transform uses the full OpenMP team internally. Keep working sets
    // bounded by processing ciphertexts sequentially rather than nesting teams.
    for(size_t row=0;row<x_cts.size();++row) {
        y_cts[row].resize(x_cts[row].size());
        for(size_t block=0;block<x_cts[row].size();++block) {
            const auto& input=x_cts[row][block];
            if(!input) throw std::invalid_argument("bootstrap: null ciphertext");
            beforeMin=std::min(beforeMin,input->GetLevel());beforeMax=std::max(beforeMax,input->GetLevel());
            auto depleted=input->Clone();
            const auto towers=depleted->GetElements()[0].GetNumOfElements();
            // OpenFHE returns the original if bootstrap cannot improve its tower
            // count. Explicitly deplete to two towers for the requested refresh.
            if(towers>2) depleted=CRYPTOCONTEXT->LevelReduce(depleted,nullptr,towers-2);
            auto output=CRYPTOCONTEXT->EvalBootstrap(depleted,1,0);
            if(output->GetSlots()!=input->GetSlots()) throw std::runtime_error("bootstrap changed logical slots");
            afterMin=std::min(afterMin,output->GetLevel());afterMax=std::max(afterMax,output->GetLevel());
            y_cts[row][block]=std::move(output); ++count;
        }
    }
    std::cout<<"BOOTSTRAP ciphertexts="<<count<<" before_level_min="<<beforeMin<<" before_level_max="<<beforeMax
             <<" depleted_towers=2 after_level_min="<<afterMin<<" after_level_max="<<afterMax
             <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<std::endl;
}
