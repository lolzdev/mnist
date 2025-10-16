import mlx.core as mx
import numpy as np
import struct

# Data loading functions. To download the data run the data.sh script.
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
        array = np.array(images / 255.0)
        return np.reshape(array, (array.shape[0], array.shape[1] * array.shape[2]))

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Softmax distribution function. Given a vector,
# the function gives another vector where the sum
# of each element equals 1.0. In this case the
# softmax is stabilized, to prevent getting an
# overflow, by subtracting the maximum value of
# the vector to each one of its elements. This
# will get all negative values without changing
# the end result since each element is getting
# shifted by the same amount.
def softmax(x):
    stable_x = x - mx.max(x, axis=1, keepdims=True)
    e = mx.exp(stable_x)
    return e / mx.sum(e, axis=1, keepdims=True)

# ReLU activation function. If x > 0 returns
# x, otherwise returns 0.0. This is needed
# to make the neural network non-linear.
def relu(x):
    return mx.maximum(0.0, x)

# Linear prediction function. Given input, weights
# and bias it computes the prediction. Should be
# activated using ReLU before passing it to the
# next layer or distributed with softmax when
# using it in the output layer.
def linear(x, W, b):
    return x @ W + b
