#pragma once
#include "layer.hpp"

// Sparse row packing, using the same decrypted FHEW sign masks as Relu_ss.
// Scheme switching must be initialized for packing_slots before forward().
class sparse_activation : public Layer {
public:
    sparse_activation(PLayerType layer_type, std::string layer_name,
                      lbcrypto::LWEPrivateKey fhe_key, uint32_t ring_dim,
                      uint32_t packing_slots);
    struct Profile {
        double conversion_ms = 0;
        double sign_ms = 0;
        double mask_ms = 0;
        size_t converted_values = 0;
        size_t evaluated_signs = 0;
    };
    const Profile& last_profile() const { return profile_; }
    using Layer::forward;
    using Layer::forward_C;
    void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
                 types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;
    void forward_C(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&,
                   types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) override;
private:
    Profile profile_;
    lbcrypto::LWEPrivateKey fhe_key_;
    uint32_t ring_dim_;
    uint32_t packing_slots_;
};
