#pragma once

#include "layer.hpp"
#include <chrono>


class Network {
public:
  Network();
  ~Network();

  types::double3d predict(types::ciphertext1d x_cts);

  void predict_P(types::ciphertext1d x_cts, types::double3d x_pts);
  
  int predict_P(types::vector2d<Ciphertext<DCRTPoly>> x_cts, types::double3d x_pts);

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};


