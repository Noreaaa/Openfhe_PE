#pragma once

#include <memory>

#include "openfhe.h"
#include "../utils/globals.hpp"
#include "../utils/types.hpp"
#include "../utils/helper.hpp"

using types::double3d;
using std::vector;
using namespace lbcrypto;


enum PLayerType {
  CONV_2D_C,
  CONV_2D,
  CONV_2D_BN,
  COMBINE_OUTPUT,
  SQUARE_ACTIVATION,
  RELU_SS_ACTIVATION,
  RELU_APPX_ACTIVATION,
  SUM_POOLING,
  AVG_POOLING,
  BOOTSTRAP,
  LINEAR,
};


class Layer {
public:
  Layer( PLayerType layer_type,
         std::string layer_name);
  Layer();
  virtual ~Layer();

  PLayerType layer_type() const { return layer_type_; };
  std::string layer_name() const { return layer_name_; };

  virtual void forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
            std::vector<double>& x_pts,
            std::vector<Ciphertext<DCRTPoly>>& y_cts,
            std::vector<double>& y_pts) {
    std::cerr <<  "forward is not implemented." << std::endl;
  }


  virtual void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts,
    double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts,
    double3d& y_pts) {
    std::cerr <<  "forward is not implemented." << std::endl;
  }

  virtual void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, types::double3d & x_pts,
    vector<Ciphertext<DCRTPoly>>& y_cts, vector<double>& y_pts) {
    std::cerr <<  "forward is not implemented." << std::endl;
  }

  virtual void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    vector<double>& y_pts) {
      std::cerr <<  "forward is not implemented." << std::endl;
    }

  virtual void forward(vector<double>& x_pts,
    vector<double>& y_pts) {
    std::cerr <<  "forward is not implemented." << std::endl;
  }


  virtual void forward(vector<Ciphertext<DCRTPoly>>& x_cts,
    vector<Ciphertext<DCRTPoly>>& y_cts, double3d& x_pts, double3d& y_pts) {
      std::cerr <<  "forward is not implemented." << std::endl;
  }


  virtual void forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
                       std::vector<Ciphertext<DCRTPoly>>& y_cts) {
    std::cerr <<  "forward is not implemented." << std::endl;
  }

  virtual void forward(Ciphertext<DCRTPoly>& x_ct,
    Ciphertext<DCRTPoly>& y_ct) {
    std::cerr << "forward is not implemented." << std::endl;
  }

  virtual void forward_C(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
  types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    std::cerr << "forward_C is not implemented." << std::endl;
  }

protected:
  PLayerType layer_type_;
  std::string layer_name_;
};


