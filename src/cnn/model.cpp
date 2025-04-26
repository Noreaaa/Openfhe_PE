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

void Network::predict_P(types::vector2d<Ciphertext<DCRTPoly>> x_cts, types::double3d x_pts){
	types::vector2d<Ciphertext<DCRTPoly>> y_cts;
	types::double3d y_pts;

	for (std::shared_ptr<Layer> layer : layers_) {
		switch (layer->layer_type()) {
		  case CONV_2D:
		  case SQUARE_ACTIVATION:
		  case AVG_POOLING:
			std::cout << "processing forwarding: " << std::endl;
	
			layer->forward(x_cts, x_pts, y_cts, y_pts);
	
			std::cout << "finish the forward pass" << std::endl;
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


}
