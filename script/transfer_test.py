import numpy as np

file_path = "../data/x_test.bin"

data = np.fromfile(file_path, dtype=np.float32)

bias = 1 * 28 * 28

for i in range(0, 28):
	for j in range(0, 28):
		print(1 if data[i*28+j + bias] > 0 else 0, end=" ")
	print()