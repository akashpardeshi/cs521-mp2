import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    # i.e., out_width <= 512
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions
    X_flat = X.reshape((batch_size, in_channels, input_height * input_width))
    M = out_channels
    TILE_M = 128

    N = out_height * out_width
    TILE_N = out_width

    K = in_channels
    TILE_K = 128

    # Process the images in batches
    print(X.shape, W.shape)
    for b in nl.affine_range(batch_size):
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                for k in nl.affine_range(K // TILE_K):
                    W_tile = nl.ndarray((TILE_M, TILE_K, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    X_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)
                    bias_tile = nl.ndarray((TILE_M, 1), dtype=bias.dtype, buffer=nl.sbuf)
                    W_tile[...] = nl.load(W[m * TILE_M:(m + 1) * TILE_M, k * TILE_K:(k + 1) * TILE_K, :, :])
                    X_tile[...] = nl.load(X_flat[b, k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                    bias_tile[...] = nl.load(bias[m * TILE_M:(m + 1) * TILE_M])

                    res_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            res_psum += nl.matmul(W_tile[:, :, i, j], X_tile)
                    res_sbuf = nl.copy(res_psum, dtype=X_out.dtype)
                    res_bias = nl.add(res_sbuf, bias_tile)
                    nl.store(X_out[b, m * TILE_M:(m + 1) * TILE_M, n, :], value=res_bias)
    return X_out

