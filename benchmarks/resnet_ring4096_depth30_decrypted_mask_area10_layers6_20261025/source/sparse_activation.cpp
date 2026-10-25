#include "sparse_activation.hpp"
#include "encrypted_relu.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

sparse_activation::sparse_activation(PLayerType layer_type, std::string layer_name,
    lbcrypto::LWEPrivateKey /*fhe_key*/, uint32_t ring_dim, uint32_t packing_slots)
    : Layer(layer_type, std::move(layer_name)),
      ring_dim_(ring_dim), packing_slots_(packing_slots) {
    if (!packing_slots_ || (packing_slots_ & (packing_slots_ - 1)) ||
        packing_slots_ > ring_dim_ / 2)
        throw std::invalid_argument("sparse_activation: invalid ring/slot configuration");
    ++CONSUMED_LEVEL;
}

void sparse_activation::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts,
    double3d& x_pts, types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    if (!CRYPTOCONTEXT || !CCLWE ||
        ring_dim_ != CRYPTOCONTEXT->GetRingDimension())
        throw std::invalid_argument("sparse_activation: missing or inconsistent scheme-switch context");
    if (x_pts.empty() || x_pts[0].empty() || x_pts[0][0].empty())
        throw std::invalid_argument("sparse_activation: empty input");
    const size_t height = x_pts[0].size(), width = x_pts[0][0].size();
    if (width > packing_slots_)
        throw std::invalid_argument("sparse_activation: slots must hold a complete row");
    for (const auto& channel : x_pts) {
        if (channel.size() != height) throw std::invalid_argument("sparse_activation: height mismatch");
        for (const auto& row : channel)
            if (row.size() != width) throw std::invalid_argument("sparse_activation: width mismatch");
    }
    if (ENCRYPTED_HEIGHT_START < 0 || ENCRYPTED_HEIGHT_END < ENCRYPTED_HEIGHT_START ||
        ENCRYPTED_HEIGHT_END >= static_cast<int>(height) || ENCRYPTED_WIDTH_START < 0 ||
        ENCRYPTED_WIDTH_END < ENCRYPTED_WIDTH_START || ENCRYPTED_WIDTH_END >= static_cast<int>(width))
        throw std::invalid_argument("sparse_activation: invalid encrypted region");
    const size_t per = packing_slots_ / width;
    const size_t blocks = (x_pts.size() + per - 1) / per;
    if (x_cts.size() != static_cast<size_t>(ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1))
        throw std::invalid_argument("sparse_activation: ciphertext row mismatch");
    for (const auto& row : x_cts) {
        if (row.size() != blocks) throw std::invalid_argument("sparse_activation: ciphertext block mismatch");
        for (const auto& ct : row)
            if (!ct || ct->GetSlots() != packing_slots_)
                throw std::invalid_argument("sparse_activation: ciphertext slot mismatch");
    }

    y_pts = x_pts;
    for (auto& channel : y_pts) for (auto& row : channel) for (auto& value : row)
        value = std::max(value, 0.0);
    y_cts.assign(x_cts.size(), std::vector<Ciphertext<DCRTPoly>>(blocks));
    // Setup/precompute occurs before forward. OpenFHE clones its switching temporaries;
    // precomputed matrices and switching keys are read-only during these calls.
    const size_t rows = x_cts.size();
    profile_ = {};
    profile_.converted_values = rows * blocks * packing_slots_;
    profile_.evaluated_signs = rows * x_pts.size() * (ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1);
    auto start = std::chrono::steady_clock::now();
    // Keep ciphertext tasks sequential so OpenFHE can use all available threads
    // internally, without nested teams or one reverse transform per CPU in memory.
    for (size_t task = 0; task < rows * blocks; ++task) {
        auto t = std::chrono::steady_clock::now();
        auto amplified = CRYPTOCONTEXT->EvalMult(x_cts[task / blocks][task % blocks], 1000.0);
        auto t1 = std::chrono::steady_clock::now();
        auto values = CRYPTOCONTEXT->EvalCKKStoFHEW(amplified, packing_slots_);
        auto t2 = std::chrono::steady_clock::now();
        std::vector<size_t> active;
        const size_t block = task % blocks;
        for (size_t channel = block * per; channel < std::min(x_pts.size(), (block+1)*per); ++channel)
            for (int w = ENCRYPTED_WIDTH_START; w <= ENCRYPTED_WIDTH_END; ++w)
                active.push_back((channel % per)*width+w);
        EncryptedReluProfile detail;
        auto mask = EncryptedReluMask(CRYPTOCONTEXT, CCLWE, values, active, packing_slots_, &detail);
        auto t3 = std::chrono::steady_clock::now();
        auto result = CRYPTOCONTEXT->EvalMult(x_cts[task / blocks][task % blocks], mask);
        if (RESCALE_REQUIRED) result = CRYPTOCONTEXT->Rescale(result);
        y_cts[task / blocks][task % blocks] = result;
        auto t4 = std::chrono::steady_clock::now();
        profile_.amplify_ms += std::chrono::duration<double,std::milli>(t1-t).count();
        profile_.conversion_ms += std::chrono::duration<double,std::milli>(t2-t1).count();
        profile_.sign_ms += detail.sign_ms;
        profile_.reverse_ms += detail.reverse_ms;
        profile_.complement_ms += detail.complement_ms;
        profile_.mask_ms += std::chrono::duration<double,std::milli>(t4-t3).count();
    }
    profile_.total_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();

}

void sparse_activation::forward_C(types::vector2d<Ciphertext<DCRTPoly>>&,
    double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) {
    throw std::logic_error("sparse_activation: compact packing is not implemented");
}
