/*
  Wevel 1.0 - Stockfish 11 derivative
  Copyright (C) 2026 Fatih W

  NNUE evaluation infrastructure for HalfKP-256 architecture.
  Implements the network topology:
    HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1
*/

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <cstdint>
#include <cstring>
#include <algorithm>

// Architecture constants for HalfKP-256-32-32-1
namespace Eval::NNUE {

constexpr int FtInDims  = 41024;  // 64 king_sq * 641 (10 piece-sq combos * 64 + 1)
constexpr int FtOutDims = 256;    // Feature transformer output per side
constexpr int L1Dims    = 32;
constexpr int L2Dims    = 32;
constexpr int PS_NB     = 641;    // Piece-square combos (+1 for alignment)

// Quantization: FT weights are int16 scaled by 64
constexpr int FtScale   = 64;
// Hidden layer weights are int8 scaled by 64
constexpr int WeightScale = 64;
// Final output divisor to get centipawn score
constexpr int FVScale   = 16;

// Alignment for SIMD operations
constexpr int SimdWidth = 32;  // 32 bytes = 256-bit AVX2

} // namespace Eval::NNUE

#endif // NNUE_COMMON_H_INCLUDED
