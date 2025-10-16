import mlx.core as mx
import numpy as np
# Functions shared by both the training and the testing scripts.
from common import *

# Hyperparameters.

HIDDEN_SIZE = 256
HIDDEN_LAYERS = 4

# Testing images. It's a numpy array of size 10000x784.
testing = load_images("dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
labels = load_labels("dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

# Load parameters from training results.
W1 = mx.array(np.loadtxt("weights/weights1.txt", dtype=np.float32))
b1 = mx.array(np.loadtxt("biases/biases1.txt", dtype=np.float32))

W_hidden = []
b_hidden = []
for i in range(HIDDEN_LAYERS):
    W_hidden.append(mx.array(np.loadtxt(f"weights/weights_hidden{i}.txt", dtype=np.float32)))
    b_hidden.append(mx.array(np.loadtxt(f"biases/biases_hidden{i}.txt", dtype=np.float32)))

W_out = mx.array(np.loadtxt("weights/weights3.txt", dtype=np.float32))
b_out = mx.array(np.loadtxt("biases/biases3.txt", dtype=np.float32))

images = mx.array(testing)
labels = mx.array(labels)

z1 = linear(images, W1, b1)
# Apply activation.
a1 = relu(z1)

# Hidden layers regression and activation.
z_hidden = [linear(a1, W_hidden[0], b_hidden[0])]
a_hidden = [relu(z_hidden[0])]
for l in range(1, HIDDEN_LAYERS):
    z_hidden.append(linear(a_hidden[-1], W_hidden[l], b_hidden[l]))
    a_hidden.append(relu(z_hidden[l]))

# Output layer regression.
z_out = linear(a_hidden[-1], W_out, b_out)
# Apply activation
y_hat = softmax(z_out)
# Transform the predictions to a vector of class indices containing the numerical label (0-9).
predictions = mx.argmax(y_hat, axis=1)
# Check how many predictions were correct.
correct = mx.sum(predictions == labels)

accuracy = (correct / labels.shape[0]) * 100
print(f"Accuracy: {accuracy.item():.2f}%")
