from tensorflow.keras.datasets import mnist
import numpy as np

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将数据标准化为 0-1 范围
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 保存数据到二进制文件
x_train.tofile("x_train.bin")
y_train.tofile("y_train.bin")
x_test.tofile("x_test.bin")
y_test.tofile("y_test.bin")
