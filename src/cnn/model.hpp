#pragma once

#include "layer.hpp"



class Network {
public:
  Network();
  ~Network();

  void predict(types::ciphertext1d x_cts);

  void add_layer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};


