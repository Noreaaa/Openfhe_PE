#include "model.hpp"
#include "../utils/experiment_phase.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sys/resource.h>

using namespace lbcrypto;

Network::Network(){
	layers_ = {};
}

Network::~Network(){
	layers_.clear();
}
// do a prediction 
/**
 * require input ciphertext
 * 
 */
types::double3d Network::predict(types::ciphertext1d x_cts){
	types::ciphertext1d y_cts;
  
	for (int i = 0; i < static_cast<int>(layers_.size()); i++){
		auto start = std::chrono::high_resolution_clock::now();
    	layers_[i]->forward(x_cts, y_cts);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = end - start;
		std::cout << "Layer " << i << " forward pass took " << elapsed.count() << " ms" << std::endl;

		x_cts.clear();
		x_cts.reserve(y_cts.size());
		for (auto& y_ct : y_cts) {
		  x_cts.push_back(y_ct);
		}
    }

	//#define DEBUG
	//#ifdef DEBUG
	//std::cout << "check cts:" << std::endl;
	types::double3d result;
	result.resize(x_cts.size());
	for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
		Plaintext plain;
		CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i], &plain);
		std::vector<double> vals = plain->GetRealPackedValue();
		for (int j = 0; j < static_cast<int>(VALID_INDEX_MAP.size()); j++){
			result[i].resize(VALID_INDEX_MAP.size());
			for (int k = 0; k < static_cast<int>(VALID_INDEX_MAP[j].size()); k++){
				result[i][j].resize(VALID_INDEX_MAP[j].size());
				int index = VALID_INDEX_MAP[j][k];
				result[i][j][k] = vals[index];
				#define DEBUG
				#ifdef DEBUG
				if (std::abs(vals[index]) < 1e-8){
					std::cout << 0 << " ";
				}
				else {
					std::cout << vals[index] << " ";
				}
				#endif		
			}
			#ifdef DEBUG
			std::cout << std::endl;
			#endif
		}
		#ifdef DEBUG
		std::cout << std::endl;
		#endif
	}

	//#endif

	return result;
}


void Network::predict_P(types::ciphertext1d x_cts, types::double3d x_pts){
	types::ciphertext1d y_cts;
	types::double3d y_pts;
	for (int i = 0; i < static_cast<int>(layers_.size()); i++){
    	layers_[i]->forward(x_cts, y_cts, x_pts, y_pts);
    }


	
	
}

int Network::predict_P(types::vector2d<Ciphertext<DCRTPoly>> x_cts, types::double3d x_pts,
    const std::string& profileCsv, int completedConvLayers) {
    types::vector2d<Ciphertext<DCRTPoly>> y_cts;
    types::double3d y_pts;
    std::vector<double> x_flat,y_flat;
    std::ofstream csv,bootstrapCsv;
    if(!profileCsv.empty()) {
        csv.open(profileCsv);
        if(!csv) throw std::runtime_error("Cannot open profile CSV: "+profileCsv);
        csv<<"conv_layers,conv_ms,relu_ms,cumulative_conv_ms,cumulative_relu_ms,total_ms,channels,height,width,encrypted_h_start,encrypted_h_end,encrypted_w_start,encrypted_w_end,ciphertexts,level,remaining_towers,slots,peak_mib,cumulative_bootstrap_ms\n";
        csv<<std::setprecision(12);
        bootstrapCsv.open(profileCsv+".bootstrap.csv");
        if(!bootstrapCsv) throw std::runtime_error("Cannot open bootstrap CSV");
        bootstrapCsv<<"after_conv_layers,bootstrap_ms,before_level,after_level,remaining_towers,ciphertexts\n";
        bootstrapCsv<<std::setprecision(12);
    }
    int convCount=completedConvLayers;
    double convMs=0,totalConv=0,totalRelu=0,totalBootstrap=0;
    for(const auto& layer:layers_) {
        const auto type=layer->layer_type();
        const int phaseLayer=type==CONV_2D?convCount+1:convCount;
        WriteExperimentPhase(profileCsv,type==CONV_2D?"server_conv":
            (type==RELU_SS_ACTIVATION || type==RELU_APPX_ACTIVATION)?"server_relu":
            type==BOOTSTRAP?"server_bootstrap":"server_other",phaseLayer);
        auto start=std::chrono::steady_clock::now();
        const auto beforeLevel=x_cts.empty()||x_cts[0].empty()?0:x_cts[0][0]->GetLevel();
        if(type==LINEAR) {
            if(layer==layers_.back()) layer->forward(x_flat,y_flat);
            else { layer->forward(x_cts,x_pts,y_flat); x_flat=std::move(y_flat); }
        } else {
            if(USE_COMPACT && (type==CONV_2D || type==RELU_SS_ACTIVATION))
                layer->forward_C(x_cts,x_pts,y_cts,y_pts);
            else layer->forward(x_cts,x_pts,y_cts,y_pts);
            x_cts=std::move(y_cts); x_pts=std::move(y_pts);
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        WriteExperimentPhase(profileCsv,"between_layers",phaseLayer);
        std::cout<<"Layer "<<layer->layer_name()<<" forward pass took "<<ms<<" ms"<<std::endl;
        if(type==BOOTSTRAP) {
            totalBootstrap+=ms;
            size_t count=0;for(const auto& row:x_cts) count+=row.size();
            if(bootstrapCsv) {
                const auto ct=x_cts.at(0).at(0);
                bootstrapCsv<<convCount<<','<<ms<<','<<beforeLevel<<','<<ct->GetLevel()<<','
                            <<ct->GetElements()[0].GetNumOfElements()<<','<<count<<'\n';bootstrapCsv.flush();
            }
        }
        if(type==CONV_2D) { ++convCount; convMs=ms; totalConv+=ms; }
        if(type==RELU_SS_ACTIVATION || type==RELU_APPX_ACTIVATION) {
            totalRelu+=ms;
            rusage usage{}; getrusage(RUSAGE_SELF,&usage);
            const double peak=usage.ru_maxrss/1024.0;
            size_t count=0; for(const auto& row:x_cts) count+=row.size();
            const auto ct=x_cts.empty() || x_cts[0].empty() ? Ciphertext<DCRTPoly>{} : x_cts[0][0];
            const auto level=ct ? ct->GetLevel() : 0;
            const auto towers=ct ? ct->GetElements()[0].GetNumOfElements() : 0;
            std::cout<<"PREFIX layers="<<convCount<<" conv_ms="<<convMs<<" relu_ms="<<ms
                     <<" cumulative_conv_ms="<<totalConv<<" cumulative_relu_ms="<<totalRelu
                     <<" total_ms="<<totalConv+totalRelu+totalBootstrap<<" output="<<x_pts.size()<<'x'<<x_pts[0].size()<<'x'<<x_pts[0][0].size()
                     <<" encrypted_h="<<ENCRYPTED_HEIGHT_START<<':'<<ENCRYPTED_HEIGHT_END
                     <<" encrypted_w="<<ENCRYPTED_WIDTH_START<<':'<<ENCRYPTED_WIDTH_END
                     <<" level="<<level<<" remaining_towers="<<towers<<" peak_mib="<<peak<<std::endl;
            if(csv) {
                csv<<convCount<<','<<convMs<<','<<ms<<','<<totalConv<<','<<totalRelu<<','<<totalConv+totalRelu+totalBootstrap<<','
                   <<x_pts.size()<<','<<x_pts[0].size()<<','<<x_pts[0][0].size()<<','
                   <<ENCRYPTED_HEIGHT_START<<','<<ENCRYPTED_HEIGHT_END<<','<<ENCRYPTED_WIDTH_START<<','<<ENCRYPTED_WIDTH_END<<','
                   <<count<<','<<level<<','<<towers<<','<<(ct ? ct->GetSlots() : 0)<<','<<peak<<','<<totalBootstrap<<'\n';
                csv.flush();
            }
            if(feature_observer_) {
                WriteExperimentPhase(profileCsv,"verification",convCount);
                feature_observer_(convCount,x_cts,x_pts);
                WriteExperimentPhase(profileCsv,"between_layers",convCount);
            }
        }
    }
    WriteExperimentPhase(profileCsv,"finished",convCount);
    if(y_flat.empty()) return -1; // This convolution prefix has no classification head.
    return std::max_element(y_flat.begin(),y_flat.end())-y_flat.begin();
}
