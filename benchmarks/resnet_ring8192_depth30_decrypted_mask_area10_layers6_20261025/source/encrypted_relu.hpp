#pragma once
#include "openfhe.h"
#include <chrono>
#include <stdexcept>
#include <vector>

// EvalSign(..., true) uses the scheme-switch-compatible q/4 sign encoding.
// Its encrypted negative indicator is converted using OpenFHE's comparison
// parameters (p=4, pmin=-1, pmax=1); 1-indicator is the positive mask.
struct EncryptedReluProfile {
    double sign_ms = 0, reverse_ms = 0, complement_ms = 0;
};
inline lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptedReluMask(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::shared_ptr<lbcrypto::BinFHEContext>& lweContext,
    const std::vector<lbcrypto::LWECiphertext>& input,
    const std::vector<size_t>& activeSlots, uint32_t slots,
    EncryptedReluProfile* profile = nullptr, bool clearInactive = false) {
    using namespace lbcrypto;
    using Clock = std::chrono::steady_clock;
    if (activeSlots.empty() || input.size() < slots)
        throw std::invalid_argument("EncryptedReluMask: empty active set or incomplete input");
    std::vector<LWECiphertext> signs(slots);
    auto start = Clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < activeSlots.size(); ++i)
        signs[activeSlots[i]] = lweContext->EvalSign(input.at(activeSlots[i]), true);
    // Inactive input slots are zero. Reusing any encrypted sign there avoids
    // extra sign evaluations; multiplication by the original zero clears them.
    const auto filler = signs[activeSlots.front()];
    for (auto& sign : signs) if (!sign) sign = filler;
    auto signedAt = Clock::now();
    auto negative = cc->EvalFHEWtoCKKS(signs, slots, slots, 4, -1.0, 1.0, 0);
    auto convertedAt = Clock::now();
    auto positive = cc->EvalSub(1.0, negative);
    if (clearInactive) {
        std::vector<double> selector(slots,0.0);
        for (auto slot : activeSlots) selector.at(slot)=1.0;
        auto plain=cc->MakeCKKSPackedPlaintext(selector,1,0,nullptr,slots);
        positive=cc->EvalMult(positive,plain);
    }
    auto end = Clock::now();
    if (profile) {
        profile->sign_ms = std::chrono::duration<double,std::milli>(signedAt-start).count();
        profile->reverse_ms = std::chrono::duration<double,std::milli>(convertedAt-signedAt).count();
        profile->complement_ms = std::chrono::duration<double,std::milli>(end-convertedAt).count();
    }
    return positive;
}
