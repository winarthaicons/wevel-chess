/*
  Wevel 1.0 - Affine transform (fully-connected) layer for NNUE.

  Computes y = W * x + b using int8 weights and int32 accumulators.
  Input x is in [0, 127] (int8 after ClippedReLU).
  Output y is int32 before activation.
*/

#ifndef AFFINE_TRANSFORM_H_INCLUDED
#define AFFINE_TRANSFORM_H_INCLUDED

#include <cstdint>
#include <cstring>

// Compute: out[i] = bias[i] + sum_j(weight[i*InDims + j] * input[j])
// Weights are stored row-major: weight[output_neuron][input_neuron]
template<int OutDims, int InDims>
inline void affine_transform(
    int32_t*       out,
    const int32_t* bias,
    const int8_t*  weight,
    const int8_t*  input)
{
    for (int i = 0; i < OutDims; ++i) {
        int32_t sum = bias[i];
        const int8_t* w = weight + i * InDims;
        for (int j = 0; j < InDims; ++j)
            sum += (int32_t)w[j] * (int32_t)input[j];
        out[i] = sum;
    }
}

// Compute: out[i] = clamp(in[i] / scale, 0, 127) cast to int8
// Used after affine_transform to prepare input for the next layer.
// scale is typically WeightScale * FtScale for the first hidden layer.
template<int Dims>
inline void clipped_relu(int8_t* out, const int32_t* in, int32_t scale) {
    for (int i = 0; i < Dims; ++i) {
        int32_t v = in[i] / scale;
        out[i] = (int8_t)(v < 0 ? 0 : v > 127 ? 127 : v);
    }
}

#endif // AFFINE_TRANSFORM_H_INCLUDED
