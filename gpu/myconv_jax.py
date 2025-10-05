import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # Convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    blocks = []
    for row in range(0, out_h, S):
        for col in range(0, out_w, S):
            block = x_pad[:, :, row:row+KH, col:col+KW].reshape((N, C*KH*KW))
            blocks.append(block)
    patches = jnp.stack(blocks, axis=1)
    return patches

def tiled_matmul(A, B, M, K, N):
    """Tiled matmul kernel"""
    C = jnp.zeros((M, N))
    Tsize = 32
    for ii in range(0, M, Tsize):
        max_i = min(ii+Tsize, M)
        for jj in range(0, N, Tsize):
            max_j = min(jj+Tsize, N)
            for kk in range(0, K, Tsize):
                max_k = min(kk+Tsize, K)
                tileA = A[ii:max_i, kk:max_k]
                tileB = B[kk:max_k, jj:max_j]
                tileC = jnp.matmul(tileA, tileB)
                C = C.at[ii:max_i, jj:max_j].add(tileC)
    return C

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = ((H + 2 * padding - (KH - 1) - 1) // stride) + 1
    out_w = ((W + 2 * padding - (KW - 1) - 1) // stride) + 1
    
    # 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # 2) flatten weight into shape (C_out, C*KH*KW).
    weight = weight.reshape((C_out, C*KH*KW))

    # 3) perform tiled matmul after required reshaping is done.
    weight = weight.T
    # out = cols @ weight
    m, k, n = out_h*out_w, C*KH*KW, C_out
    out = jnp.empty((N, m, n))
    for i in range(N):
        out = out.at[i, :, :].set(tiled_matmul(cols[i, :, :], weight, m, k, n))

    # 4) Add bias.
    out += bias

    # 5) reshape output into shape (N, C_out, out_h, out_w).
    out = jnp.permute_dims(out, (0, 2, 1)).reshape(N, C_out, out_h, out_w)

    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    # jax.profiler.start_trace(log_dir=logdir)
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
    # jax.profiler.stop_trace()

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
