#include "model.hpp"

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

int Network::predict_P(types::vector2d<Ciphertext<DCRTPoly>> x_cts, types::double3d x_pts){
	types::vector2d<Ciphertext<DCRTPoly>> y_cts;
	types::double3d y_pts;
	vector<double> x_pts_1d;
	vector<double> y_pts_1d;

	for (std::shared_ptr<Layer> layer : layers_) {
		auto start = std::chrono::high_resolution_clock::now();
		switch (layer->layer_type()) {
		case LINEAR:
			if(layer == layers_.back()){
				layer->forward(x_pts_1d, y_pts_1d);
			}
			else{
				layer->forward(x_cts, x_pts, y_pts_1d);
			x_pts_1d = std::move(y_pts_1d);
			}
		break;
		case CONV_2D:
		case RELU_SS_ACTIVATION:
			if (USE_COMPACT==true){
				layer->forward_C(x_cts, x_pts, y_cts, y_pts);
			}
			else {
				layer->forward(x_cts, x_pts, y_cts, y_pts);
			}

			x_cts.clear();
			x_cts.reserve(y_cts.size());
			for (auto& y_ct : y_cts) {
			  x_cts.push_back(y_ct);
			}
			x_pts = std::move(y_pts);
		break;
		case SQUARE_ACTIVATION:
		case AVG_POOLING:
		case BOOTSTRAP:
		case RELU_APPX_ACTIVATION:

		layer->forward(x_cts, x_pts, y_cts, y_pts);

		x_cts.clear();
		x_cts.reserve(y_cts.size());
		for (auto& y_ct : y_cts) {
		  x_cts.push_back(y_ct);
		}
		x_pts = std::move(y_pts);
		break;
		default:
		break;

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = end - start;
		std::cout << "Layer " << layer->layer_name() << " forward pass took " << elapsed.count() << " ms" << std::endl;
		}
	}

	//#define DEBUG
	#ifdef DEBUG
	std::cout << "check cts:" << std::endl;
    //for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
    //    std::cout << "encrypted row: " << i << std::endl;
	//	int channel_cout = 0;
    //    for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
    //        Plaintext plain;
    //        CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &plain);
	//		std::vector<double> vals = plain->GetRealPackedValue();
	//		for (int k = 0; k < static_cast<int>(vals.size()); k++){
	//			if (channel_cout % 8 == 0){
	//				std::cout << "channel[" << channel_cout / 8 << "]: ";
	//			}
	//			if (std::abs(vals[k]) < 1e-8){
	//				std::cout << 0 << " ";
	//			}
	//			else {
	//				std::cout << vals[k] << " ";
	//			}
	//			if (channel_cout % 8 == 7){
	//				std::cout << std::endl;
	//			}
	//			channel_cout++;
	//		}
    //    }
    //}

	for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
        std::cout << "encrypted row: " << i << std::endl;
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            Plaintext plain;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &plain);
			std::vector<double> vals = plain->GetRealPackedValue();
			for (int k = 0; k < static_cast<int>(vals.size()); k++){

				if (std::abs(vals[k]) < 1e-8){
					std::cout << 0 << " ";
				}
				else {
					std::cout << vals[k] << " ";
				}
			}
        }
		std::cout << std::endl;
    }


	//std::cout << "check pts:" << std::endl;
	//print_3d(x_pts);

	//return 0;
	#endif

	int max_type = -1;
	double max = -1000000;
	for (size_t i = 0; i < y_pts_1d.size(); i++){
		std::cout << "y_pts_1d[" << i << "]: " << y_pts_1d[i] << std::endl;
		if (y_pts_1d[i] > max){
			max = y_pts_1d[i];
			max_type = i;
		}
	}
	std::cout << "predicted result: " << max_type << std::endl;

	return max_type;
	
}
