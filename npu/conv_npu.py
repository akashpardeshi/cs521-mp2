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
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    # c_in_pmax = nl.tile_size.pmax
    # n_tiles_c_in = in_channels // c_in_pmax

    M = out_channels
    N = out_height * out_width
    K = in_channels

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # Allocate tile for matmul results in psum
        out_tile = nl.zeros(shape=(M, N), dtype=nl.float32, buffer=nl.psum)
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                # Allocate tiles for X, W, and matmul output in sbuf
                W_tile = nl.ndarray(shape=(M, K), dtype=W.dtype, buffer=nl.sbuf)
                X_tile = nl.ndarray(shape=(K, N), dtype=X.dtype, buffer=nl.sbuf)

                # Load tiles
                W_tile = nl.load(W[:, :, i, j]) # TODO: transpose to (KH, KW, Cout, Cin) would be nice
                X_tile = nl.load(X[b, :, i:i+out_height, j:j+out_width])

                # Matmul
                out_tile += nl.matmul(W_tile, X_tile)

        # Copy out and store
        res = nl.copy(out_tile, dtype=X_out.dtype)
        nl.store(X_out[b, :, :, :], value=res)
    return X_out

