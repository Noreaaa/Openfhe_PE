#pragma once

#include "layer.hpp"
#include <chrono>
#include <functional>


class Network {
public:
  Network();
  ~Network();

  types::double3d predict(types::ciphertext1d x_cts);

  void predict_P(types::ciphertext1d x_cts, types::double3d x_pts);
  
  int predict_P(types::vector2d<Ciphertext<DCRTPoly>> x_cts, types::double3d x_pts, const std::string& profile_csv = "", int completed_conv_layers = 0);

  using FeatureObserver = std::function<void(int, const types::vector2d<Ciphertext<DCRTPoly>>&, const types::double3d&)>;
  void set_feature_observer(FeatureObserver observer) { feature_observer_ = std::move(observer); }

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  FeatureObserver feature_observer_;
  std::vector<std::shared_ptr<Layer>> layers_;
};


