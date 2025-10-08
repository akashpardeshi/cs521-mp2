import numpy as np
import matplotlib.pyplot as plt

def kernel_size_scaling():
    k = [3, 5, 7]
    # runtimes in ms
    pytorch = [278, 124, 124]
    inductor = [8, 11, 11]
    jax = [0.263, 0.373, 0.309]

    plt.title('Kernel Size Scaling')
    plt.xlabel('kernel size (kh = kw)')
    plt.ylabel('Execution time (ms)')
    plt.plot(k, pytorch, 'o-', label='pytorch')
    plt.plot(k, inductor, 'o-', label='inductor')
    plt.plot(k, jax, 'o-', label='jax')
    plt.legend()
    plt.show()

def input_size_scaling():
    hw = [8, 16, 24]
    # runtimes in ms
    pytorch = [111, 124, 143]
    inductor = [2, 11, 22]
    jax = [0.125, 0.373, None]

    plt.title('Input Size Scaling')
    plt.xlabel('input size (h = w)')
    plt.ylabel('Execution time (ms)')
    plt.plot(hw, pytorch, 'o-', label='pytorch')
    plt.plot(hw, inductor, 'o-', label='inductor')
    plt.plot(hw, jax, 'o-', label='jax')
    plt.legend()
    plt.show()

kernel_size_scaling()
# input_size_scaling()
