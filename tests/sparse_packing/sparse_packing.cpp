#include "openfhe.h"
#include "binfhecontext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

struct Options {
    uint32_t ring = 16384, slots = 1024, outputs = 64;
    uint32_t depth = 3, level = 0, scaleBits = 50, firstBits = 60;
    uint32_t repeats = 5, warmup = 1, threads = 1;
};

static uint32_t parseUint(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer: " + value);
    auto n = std::stoull(value);
    if (n > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Integer too large: " + value);
    return static_cast<uint32_t>(n);
}

static bool powerOfTwo(uint32_t n) { return n && !(n & (n - 1)); }
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--help") {
                std::cout << "Usage: sparse_packing_bench [--ring 16384] [--slots 1024]\n"
                          << "  [--outputs 64] [--repeats 5] [--warmup 1] [--threads 1]\n"
                          << "  [--depth 3] [--level 0] [--scale-bits 50] [--first-bits 60]\n"
                          << "CKKS: HEStd_NotSet, UNIFORM_TERNARY, HYBRID. FHEW: STD128.\n"
                          << "Times are milliseconds. stdout is CSV; diagnostics go to stderr.\n";
                return 0;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + arg);
            uint32_t value = parseUint(argv[++i]);
            if (arg == "--ring") o.ring = value;
            else if (arg == "--slots") o.slots = value;
            else if (arg == "--outputs") o.outputs = value;
            else if (arg == "--repeats") o.repeats = value;
            else if (arg == "--warmup") o.warmup = value;
            else if (arg == "--threads") o.threads = value;
            else if (arg == "--depth") o.depth = value;
            else if (arg == "--level") o.level = value;
            else if (arg == "--scale-bits") o.scaleBits = value;
            else if (arg == "--first-bits") o.firstBits = value;
            else throw std::invalid_argument("Unknown option: " + arg);
        }
        if (!powerOfTwo(o.ring) || !powerOfTwo(o.slots) || o.slots > o.ring / 2)
            throw std::invalid_argument("ring and slots must be powers of two; slots <= ring/2");
        if (!o.outputs || o.outputs > o.slots || !o.repeats || !o.threads ||
            o.threads > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Require 1 <= outputs <= slots, repeats > 0, valid threads > 0");
        if (o.depth < 3 || o.level > o.depth - 3)
            throw std::invalid_argument("Leave at least depth 3 for switching: level <= depth - 3");
        if (o.scaleBits < 30 || o.scaleBits > 59 || o.firstBits < o.scaleBits || o.firstBits > 60)
            throw std::invalid_argument("Require 30 <= scale-bits <= 59 and scale-bits <= first-bits <= 60");

        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(o.threads));
        std::cerr << "OpenFHE=" << BENCH_OPENFHE_VERSION << ", NATIVEINT=" << NATIVEINT
                  << ", ring=" << o.ring << ", slots=" << o.slots
                  << ", outputs=" << o.outputs << ", threads=" << omp_get_max_threads()
                  << ", CKKS=HEStd_NotSet, FHEW=STD128\n";

        auto setupStart = Clock::now();
        CCParams<CryptoContextCKKSRNS> p;
        p.SetSecurityLevel(HEStd_NotSet);
        p.SetRingDim(o.ring);
        p.SetBatchSize(o.slots);
        p.SetMultiplicativeDepth(o.depth);
        p.SetScalingModSize(o.scaleBits);
        p.SetFirstModSize(o.firstBits);
        p.SetSecretKeyDist(UNIFORM_TERNARY);
        p.SetKeySwitchTechnique(HYBRID);
        p.SetNumLargeDigits(3);
#if NATIVEINT == 128
        p.SetScalingTechnique(FIXEDAUTO);
#else
        p.SetScalingTechnique(FLEXIBLEAUTO);
#endif
        auto cc = GenCryptoContext(p);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(SCHEMESWITCH);
        auto keys = cc->KeyGen();

        constexpr uint32_t logQ = 25;
        SchSwchParams sw;
        sw.SetSecurityLevelCKKS(HEStd_NotSet);
        sw.SetSecurityLevelFHEW(STD128);
        sw.SetNumSlotsCKKS(o.slots);
        sw.SetNumValues(o.outputs);
        sw.SetCtxtModSizeFHEWLargePrec(logQ);
        auto lweKey = cc->EvalCKKStoFHEWSetup(sw);
        cc->EvalCKKStoFHEWKeyGen(keys, lweKey);
        auto lwe = cc->GetBinCCForSchemeSwitch();
        const uint64_t plaintextModulus = (uint64_t{1} << logQ) /
                                         (2 * lwe->GetBeta().ConvertToInt());
        if (plaintextModulus < 64)
            throw std::runtime_error("Unexpectedly small LWE plaintext modulus");
        cc->EvalCKKStoFHEWPrecompute(1.0 / static_cast<double>(plaintextModulus));
        double setupMs = elapsed(setupStart);

        // Explicit slots select genuine sparse encoding, not just full packing with zeros.
        // The first K values are identical for every S, including negative values and zero.
        std::vector<double> values(o.slots);
        for (uint32_t i = 0; i < o.slots; ++i)
            values[i] = static_cast<int>(i % 17) - 8;
        auto pt = cc->MakeCKKSPackedPlaintext(values, 1, o.level, nullptr, o.slots);
        auto ct = cc->Encrypt(keys.publicKey, pt);
        std::cerr << "actual_ring=" << cc->GetRingDimension() << ", input_level=" << ct->GetLevel()
                  << ", input_towers=" << ct->GetElements()[0].GetNumOfElements()
                  << ", lwe_n=" << lwe->GetParams()->GetLWEParams()->Getn()
                  << ", pLWE=" << plaintextModulus << ", setup_ms=" << setupMs << '\n';

        Plaintext decoded;
        cc->Decrypt(keys.secretKey, ct, &decoded);
        decoded->SetLength(o.slots);
        auto unpacked = decoded->GetRealPackedValue();
        double ckksError = 0;
        for (uint32_t i = 0; i < o.slots; ++i) {
            if (!std::isfinite(unpacked.at(i)))
                throw std::runtime_error("Non-finite CKKS decoded value");
            ckksError = std::max(ckksError, std::abs(unpacked.at(i) - values[i]));
        }
        if (ckksError > 0.1)
            throw std::runtime_error("CKKS input check failed");

        auto check = [&](const std::vector<LWECiphertext>& result) {
            if (result.size() != o.outputs)
                throw std::runtime_error("Unexpected LWE output count");
            uint64_t maxError = 0;
            const int64_t modulus = static_cast<int64_t>(plaintextModulus);
            for (uint32_t i = 0; i < o.outputs; ++i) {
                LWEPlaintext actual;
                lwe->Decrypt(lweKey, result[i], &actual, plaintextModulus);
                const int64_t expected = static_cast<int64_t>(std::llround(values[i]));
                const int64_t residue = ((static_cast<int64_t>(actual) - expected) % modulus + modulus) % modulus;
                maxError = std::max(maxError, static_cast<uint64_t>(std::min(residue, modulus - residue)));
            }
            return maxError;
        };

        // Clone outside the timed region so every call gets the same input state.
        for (uint32_t i = 0; i < o.warmup; ++i) {
            auto input = ct->Clone();
            if (check(cc->EvalCKKStoFHEW(input, o.outputs)) > 1)
                throw std::runtime_error("Warmup conversion check failed (error > 1)");
        }
        std::vector<double> timings;
        uint64_t worstError = 0;
        for (uint32_t i = 0; i < o.repeats; ++i) {
            auto input = ct->Clone();
            auto start = Clock::now();
            auto result = cc->EvalCKKStoFHEW(input, o.outputs);
            double ms = elapsed(start);
            timings.push_back(ms);
            auto error = check(result);  // Decryption and validation are NOT timed.
            worstError = std::max(worstError, error);
            std::cerr << "trial=" << i + 1 << ", switch_ms=" << ms
                      << ", max_modular_error=" << error << '\n';
        }
        std::sort(timings.begin(), timings.end());
        const size_t middle = timings.size() / 2;
        double median = timings.size() % 2 ? timings[middle] :
                        (timings[middle - 1] + timings[middle]) / 2;
        double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        std::cout << "openfhe,ring,slots,outputs,depth,level,scale_bits,first_bits,threads,repeats,"
                     "warmup,setup_ms,min_ms,median_ms,mean_ms,max_ms,ckks_max_error,lwe_max_error,passed\n";
        std::cout << std::setprecision(10) << BENCH_OPENFHE_VERSION << ',' << cc->GetRingDimension()
                  << ',' << o.slots << ',' << o.outputs << ',' << o.depth << ',' << o.level
                  << ',' << o.scaleBits << ',' << o.firstBits << ',' << o.threads
                  << ',' << o.repeats << ',' << o.warmup << ',' << setupMs
                  << ',' << timings.front() << ',' << median << ',' << mean << ',' << timings.back()
                  << ',' << ckksError << ',' << worstError << ',' << (worstError <= 1 ? "true" : "false") << '\n';
        return worstError <= 1 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
