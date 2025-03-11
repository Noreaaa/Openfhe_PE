#include "layer.hpp"


Layer::Layer(PLayerType layer_type,
            std::string layer_name)
    : layer_type_(layer_type), layer_name_(layer_name) {}
Layer::Layer() {}
Layer::~Layer() {}

