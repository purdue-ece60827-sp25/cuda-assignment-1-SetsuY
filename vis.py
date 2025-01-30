import matplotlib.pyplot as plt
import numpy as np

# part1
saxpy_vec_size_exp = [0, 5, 10, 15, 20, 25]
x = np.array(saxpy_vec_size_exp)
saxpy_kern = np.array([2.208, 2.400, 2.112, 2.368, 15.553, 498.364])
saxpy_malloc = np.array([105517.441, 95394.094, 107790.783, 101746.165, 97083.433, 104308.648])
saxpy_memcpy = np.array([57.155, 57.055, 57.094, 252.831, 4575.635, 104308.648])

plt.bar(x, saxpy_kern, bottom=saxpy_malloc+saxpy_memcpy, label="Kernel")
plt.bar(x, saxpy_memcpy, bottom=saxpy_malloc, label="cudaMemcpy")
plt.bar(x, saxpy_malloc, label="cudaMalloc")

plt.xlabel("log2 of Vector Size")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.show()

# part2
num_per_block_exp = np.array([0, 2, 4, 6])
gen_kern = np.array([206.975, 228.765, 2455.691, 218178.942])
reduce_kern = np.array([5.952, 5.984, 5.984, 5.536])
malloc = np.array([105491.154, 100130.762, 102536.661, 110751.233])

plt.bar(num_per_block_exp, reduce_kern, bottom=gen_kern+malloc, label="Reduce")
plt.bar(num_per_block_exp, gen_kern, bottom=malloc, label="Generate Points")
plt.bar(num_per_block_exp, malloc, label="cudaMalloc")

plt.xlabel("log10 of Points per Block")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.show()

num_blocks_exp = np.array([0, 5, 10, 15, 20])
gen_kern = np.array([130765.584, 130564.001, 202085.735, 203442.171, 5067834.447])
malloc = np.array([94664.301, 96369.986, 93306.275, 101983.750, 92118.649])

plt.bar(num_blocks_exp, gen_kern, bottom=malloc, label="Generate Points")
plt.bar(num_blocks_exp, malloc, label="cudaMalloc")

plt.xlabel("log2 of Number of Threads")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.show()
