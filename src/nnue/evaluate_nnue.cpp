/*
  Wevel 1.0 - NNUE evaluation implementation.

  Network: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1

  Binary .nnue file format (compatible with Stockfish 12/13):
    uint32_t  version         (0x7AF32F16)
    uint32_t  hash_value      (architecture hash, read for verification)
    uint32_t  arch_desc_len   (length of description string)
    char      arch_desc[]     (e.g. "Features=HalfKP...")
    int16_t   ft_biases[256]
    int16_t   ft_weights[41024 * 256]  (feature-major: all 256 outputs per feature)
    uint32_t  l1_hash
    int32_t   l1_biases[32]
    int8_t    l1_weights[32 * 512]     (output-major: all 512 inputs per output)
    uint32_t  l2_hash
    int32_t   l2_biases[32]
    int8_t    l2_weights[32 * 32]
    uint32_t  out_hash
    int32_t   out_bias[1]
    int8_t    out_weights[32]

  The feature transformer uses FtScale=64 (int16 weights × 64).
  Hidden layer weights are int8, biases are int32 pre-scaled.
  Final score: output_int32 / FVScale = centipawns (from STM perspective).
*/

#include "evaluate_nnue.h"
#include "nnue_common.h"
#include "features/half_kp.h"
#include "layers/affine_transform.h"
#include "../position.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>
#include <algorithm>
#include "../misc.h"   // for sync_cout / sync_endl

#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace Eval {

bool useNNUE = false;

namespace NNUE {

// ---- Weight storage --------------------------------------------------------

namespace {

// Feature transformer weights (heap-allocated: 41024 * 256 * 2 bytes = ~20 MB)
static int16_t* ft_weights  = nullptr;  // [FtInDims][FtOutDims]
static int16_t  ft_biases[FtOutDims];

// L1: 512 → 32
static int8_t   l1_weights[L1Dims * FtOutDims * 2];  // [L1Dims][FtOutDims*2]
static int32_t  l1_biases[L1Dims];

// L2: 32 → 32
static int8_t   l2_weights[L2Dims * L1Dims];
static int32_t  l2_biases[L2Dims];

// Output: 32 → 1
static int8_t   out_weights[L2Dims];
static int32_t  out_bias;

static bool     net_loaded = false;

} // namespace

// ---- File loading ----------------------------------------------------------

void init() {
    if (!ft_weights) {
        ft_weights = new int16_t[FtInDims * FtOutDims];
    }
    std::memset(ft_biases,   0, sizeof(ft_biases));
    std::memset(ft_weights,  0, FtInDims * FtOutDims * sizeof(int16_t));
    std::memset(l1_weights,  0, sizeof(l1_weights));
    std::memset(l1_biases,   0, sizeof(l1_biases));
    std::memset(l2_weights,  0, sizeof(l2_weights));
    std::memset(l2_biases,   0, sizeof(l2_biases));
    std::memset(out_weights, 0, sizeof(out_weights));
    out_bias    = 0;
    net_loaded  = false;
}

bool load_eval(const std::string& fname) {
    if (!ft_weights) init();

    if (fname.empty() || fname == "<empty>") {
        net_loaded = false;
        useNNUE    = false;
        return false;
    }

    std::ifstream f(fname, std::ios::binary);
    if (!f.is_open()) {
        sync_cout << "info string NNUE: cannot open " << fname << sync_endl;
        return false;
    }

    auto read32 = [&]() -> uint32_t {
        uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v;
    };

    // Header
    uint32_t version = read32();
    if (version != 0x7AF32F16u) {
        std::cerr << "NNUE: unsupported version 0x" << std::hex << version << std::endl;
        return false;
    }
    /* hash = */ read32();
    uint32_t desc_len = read32();
    f.seekg(desc_len, std::ios::cur);  // skip architecture description string

    // Feature transformer section: 4-byte section hash, then biases and weights
    /* ft_hash = */ read32();
    f.read(reinterpret_cast<char*>(ft_biases),  sizeof(ft_biases));
    f.read(reinterpret_cast<char*>(ft_weights), FtInDims * FtOutDims * sizeof(int16_t));

    // Network section: single 4-byte hash covers all FC layers that follow
    /* net_hash = */ read32();
    f.read(reinterpret_cast<char*>(l1_biases),  sizeof(l1_biases));
    f.read(reinterpret_cast<char*>(l1_weights), sizeof(l1_weights));
    f.read(reinterpret_cast<char*>(l2_biases),  sizeof(l2_biases));
    f.read(reinterpret_cast<char*>(l2_weights), sizeof(l2_weights));
    f.read(reinterpret_cast<char*>(&out_bias),  sizeof(out_bias));
    f.read(reinterpret_cast<char*>(out_weights), sizeof(out_weights));

    if (!f) {
        std::cerr << "NNUE: read error in " << fname << std::endl;
        return false;
    }

    net_loaded = true;
    useNNUE    = true;
    sync_cout << "info string NNUE: loaded " << fname << sync_endl;
    return true;
}

bool verify_net() {
    return net_loaded && ft_weights != nullptr;
}

// ---- Evaluation ------------------------------------------------------------

Value evaluate(const Position& pos) {
    if (!net_loaded) return VALUE_ZERO;

    const Color us = pos.side_to_move();

    // --- 1. Compute feature transformer accumulators for both sides ---
    // FT accumulation in int16 (int16 is sufficient: biases ~±32k, each feature
    // weight ~±100, 30 features max, so sum ~ 3000 which fits int16 easily).

    alignas(64) int16_t acc[2][FtOutDims];

    for (int persp = 0; persp < 2; ++persp) {
        int feature_indices[32];
        int n = Features::collect_features(Color(persp), pos, feature_indices);

#ifdef USE_AVX2
        // AVX2 path: process 16 int16 lanes at a time (256-bit = 16 * int16)
        constexpr int kChunkSize = 256 / 16;  // = 16
        static_assert(FtOutDims % kChunkSize == 0, "FtOutDims must be multiple of 16");

        __m256i* acc_v = reinterpret_cast<__m256i*>(acc[persp]);
        const __m256i* bias_v = reinterpret_cast<const __m256i*>(ft_biases);
        for (int j = 0; j < FtOutDims / kChunkSize; ++j)
            acc_v[j] = bias_v[j];

        for (int k = 0; k < n; ++k) {
            const __m256i* w = reinterpret_cast<const __m256i*>(
                ft_weights + feature_indices[k] * FtOutDims);
            for (int j = 0; j < FtOutDims / kChunkSize; ++j)
                acc_v[j] = _mm256_add_epi16(acc_v[j], w[j]);
        }
#else
        // Scalar fallback
        for (int j = 0; j < FtOutDims; ++j)
            acc[persp][j] = ft_biases[j];
        for (int k = 0; k < n; ++k) {
            int base = feature_indices[k] * FtOutDims;
            for (int j = 0; j < FtOutDims; ++j)
                acc[persp][j] += ft_weights[base + j];
        }
#endif
    }

    // --- 2. ClippedReLU on both accumulators, STM perspective first ---

    // Input to L1: [STM_acc(256) | NSTM_acc(256)] = 512 int8 values
    const int stm  = (us == WHITE) ? 0 : 1;
    const int nstm = 1 - stm;

    alignas(64) int8_t l1_input[FtOutDims * 2];
    for (int j = 0; j < FtOutDims; ++j) {
        int sv = acc[stm][j];
        int nv = acc[nstm][j];
        l1_input[j]             = (int8_t)(sv  < 0 ? 0 : sv  > 127 ? 127 : sv);
        l1_input[j + FtOutDims] = (int8_t)(nv  < 0 ? 0 : nv  > 127 ? 127 : nv);
    }

    // --- 3-5. FC forward pass (L1 → L2 → output) ---

#ifdef USE_AVX2
    // Vectorized L1: 512 → 32 using AVX2 maddubs/madd
    // Inputs are uint8 in [0, 127] (treat int8 as uint8 since values ≥ 0)
    // Weights are int8. Biases are int32.
    alignas(64) int8_t l2_input[L1Dims];
    {
        constexpr int kInChunk = 32;  // 256-bit AVX2 = 32 int8
        static_assert((FtOutDims * 2) % kInChunk == 0, "");

        for (int i = 0; i < L1Dims; ++i) {
            __m256i sum = _mm256_setzero_si256();
            const int8_t* w = l1_weights + i * FtOutDims * 2;
            const int8_t* x = l1_input;
            for (int j = 0; j < FtOutDims * 2; j += kInChunk) {
                __m256i xv = _mm256_loadu_si256((const __m256i*)(x + j));
                __m256i wv = _mm256_loadu_si256((const __m256i*)(w + j));
                // madd: (uint8 × int8) → int16 with horizontal pairs add
                __m256i p  = _mm256_maddubs_epi16(xv, wv);
                // madd ×1: int16 pairs → int32
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
            }
            // Horizontal sum of 8 int32s in sum
            __m128i s128 = _mm_add_epi32(_mm256_extracti128_si256(sum, 0),
                                          _mm256_extracti128_si256(sum, 1));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
            int32_t val = _mm_cvtsi128_si32(s128) + l1_biases[i];
            val /= WeightScale;
            l2_input[i] = (int8_t)(val < 0 ? 0 : val > 127 ? 127 : val);
        }
    }
#else
    int32_t l1_out[L1Dims];
    affine_transform<L1Dims, FtOutDims * 2>(l1_out, l1_biases, l1_weights, l1_input);
    alignas(64) int8_t l2_input[L1Dims];
    clipped_relu<L1Dims>(l2_input, l1_out, WeightScale);
#endif

    // L2: 32 → 32 (small, scalar is fine)
    int32_t l2_out[L2Dims];
    affine_transform<L2Dims, L1Dims>(l2_out, l2_biases, l2_weights, l2_input);
    int8_t out_input[L2Dims];
    clipped_relu<L2Dims>(out_input, l2_out, WeightScale);

    // Output: 32 → 1
    int32_t output_val = out_bias;
    for (int j = 0; j < L2Dims; ++j)
        output_val += (int32_t)out_weights[j] * (int32_t)out_input[j];

    return Value(output_val / FVScale);
}

} // namespace NNUE
} // namespace Eval
