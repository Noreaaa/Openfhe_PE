#include "baseline_sparse_activation.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

baseline_sparse_activation::baseline_sparse_activation(PLayerType layer_type, std::string layer_name,
    lbcrypto::LWEPrivateKey fhe_key, uint32_t ring_dim, uint32_t packing_slots)
    : Layer(layer_type, std::move(layer_name)), fhe_key_(std::move(fhe_key)),
      ring_dim_(ring_dim), packing_slots_(packing_slots) {
    if (!packing_slots_ || (packing_slots_ & (packing_slots_ - 1)) ||
        packing_slots_ > ring_dim_ / 2)
        throw std::invalid_argument("baseline_sparse_activation: invalid ring/slot configuration");
    ++CONSUMED_LEVEL;
}

void baseline_sparse_activation::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts,
    double3d& x_pts, types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    if (!CRYPTOCONTEXT || !CCLWE || !fhe_key_ ||
        ring_dim_ != CRYPTOCONTEXT->GetRingDimension())
        throw std::invalid_argument("baseline_sparse_activation: missing or inconsistent scheme-switch context");
    if (x_pts.empty() || x_pts[0].empty() || x_pts[0][0].empty())
        throw std::invalid_argument("baseline_sparse_activation: empty input");
    const size_t height = x_pts[0].size(), width = x_pts[0][0].size();
    if (width > packing_slots_)
        throw std::invalid_argument("baseline_sparse_activation: slots must hold a complete row");
    for (const auto& channel : x_pts) {
        if (channel.size() != height) throw std::invalid_argument("baseline_sparse_activation: height mismatch");
        for (const auto& row : channel)
            if (row.size() != width) throw std::invalid_argument("baseline_sparse_activation: width mismatch");
    }
    if (ENCRYPTED_HEIGHT_START < 0 || ENCRYPTED_HEIGHT_END < ENCRYPTED_HEIGHT_START ||
        ENCRYPTED_HEIGHT_END >= static_cast<int>(height) || ENCRYPTED_WIDTH_START < 0 ||
        ENCRYPTED_WIDTH_END < ENCRYPTED_WIDTH_START || ENCRYPTED_WIDTH_END >= static_cast<int>(width))
        throw std::invalid_argument("baseline_sparse_activation: invalid encrypted region");
    const size_t per = packing_slots_ / width;
    const size_t blocks = (x_pts.size() + per - 1) / per;
    if (x_cts.size() != static_cast<size_t>(ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1))
        throw std::invalid_argument("baseline_sparse_activation: ciphertext row mismatch");
    for (const auto& row : x_cts) {
        if (row.size() != blocks) throw std::invalid_argument("baseline_sparse_activation: ciphertext block mismatch");
        for (const auto& ct : row)
            if (!ct || ct->GetSlots() != packing_slots_)
                throw std::invalid_argument("baseline_sparse_activation: ciphertext slot mismatch");
    }

    y_pts = x_pts;
    for (auto& channel : y_pts) for (auto& row : channel) for (auto& value : row)
        value = std::max(value, 0.0);
    y_cts.assign(x_cts.size(), std::vector<Ciphertext<DCRTPoly>>(blocks));
    // Setup/precompute occurs before forward. OpenFHE clones its switching temporaries;
    // precomputed matrices and switching keys are read-only during these calls.
    const size_t rows = x_cts.size();
    std::vector<std::vector<LWECiphertext>> lwe(rows * blocks);
    std::vector<std::vector<double>> masks(rows * blocks, std::vector<double>(packing_slots_, 0.0));
    profile_ = {};
    profile_.converted_values = rows * blocks * packing_slots_;
    profile_.evaluated_signs = rows * x_pts.size() * (ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1);
    auto start = std::chrono::steady_clock::now();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t task = 0; task < rows * blocks; ++task) {
        auto amplified = CRYPTOCONTEXT->EvalMult(x_cts[task / blocks][task % blocks], 1000.0);
        // Explicit logical slot count, including the final partially filled block.
        lwe[task] = CRYPTOCONTEXT->EvalCKKStoFHEW(amplified, packing_slots_);
    }
    auto afterConversion = std::chrono::steady_clock::now();
    const size_t encryptedWidth = ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t task = 0; task < profile_.evaluated_signs; ++task) {
        const size_t h = task / (x_pts.size() * encryptedWidth);
        const size_t channel = (task / encryptedWidth) % x_pts.size();
        const size_t block = channel / per;
        const size_t slot = (channel % per) * width + ENCRYPTED_WIDTH_START + task % encryptedWidth;
        auto sign = CCLWE->EvalSign(lwe[h * blocks + block].at(slot));
        LWEPlaintext negative;
        CCLWE->Decrypt(fhe_key_, sign, &negative, 2);
        masks[h * blocks + block][slot] = negative == 0 ? 1.0 : 0.0;
    }
    auto afterSign = std::chrono::steady_clock::now();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t task = 0; task < rows * blocks; ++task) {
        auto plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(masks[task], 1, 0, nullptr, packing_slots_);
        auto& result = y_cts[task / blocks][task % blocks];
        result = CRYPTOCONTEXT->EvalMult(x_cts[task / blocks][task % blocks], plain);
        if (RESCALE_REQUIRED) result = CRYPTOCONTEXT->Rescale(result);
    }
    auto afterMask = std::chrono::steady_clock::now();
    profile_.conversion_ms = std::chrono::duration<double, std::milli>(afterConversion - start).count();
    profile_.sign_ms = std::chrono::duration<double, std::milli>(afterSign - afterConversion).count();
    profile_.mask_ms = std::chrono::duration<double, std::milli>(afterMask - afterSign).count();

}

void baseline_sparse_activation::forward_C(types::vector2d<Ciphertext<DCRTPoly>>&,
    double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) {
    throw std::logic_error("baseline_sparse_activation: compact packing is not implemented");
}
