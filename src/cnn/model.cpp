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
void Network::predict(types::ciphertext1d x_cts){
	types::ciphertext1d y_cts;
  
	for (int i = 0; i < static_cast<int>(layers_.size()); i++){
    	layers_[i]->forward(x_cts, y_cts);
    }

	
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
		  case SQUARE_ACTIVATION:
		  case AVG_POOLING:
		  case BOOTSTRAP:
	
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
		}
	}

	//#define DEBUG
	#ifdef DEBUG
	std::cout << "check cts:" << std::endl;
    for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
        std::cout << "encrypted row: " << i << std::endl;
		int channel_cout = 0;
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            Plaintext plain;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &plain);
			std::vector<double> vals = plain->GetRealPackedValue();
			for (int k = 0; k < static_cast<int>(vals.size()); k++){
				if (channel_cout % 4 == 0){
					std::cout << "channel[" << channel_cout / 16 << "]: ";
				}
				if (std::abs(vals[k]) < 1e-8){
					std::cout << 0 << " ";
				}
				else {
					std::cout << vals[k] << " ";
				}
				if (channel_cout % 4 == 3){
					std::cout << std::endl;
				}
				channel_cout++;
			}
        }
    }

	return 0;
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
