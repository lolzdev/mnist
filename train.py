import mlx.core as mx
import numpy as np
import math
# Functions shared by both the training and the testing scripts.
from common import *

# Hyperparameters

HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
EPOCHS = 10
BATCH_SIZE = 60
HIDDEN_LAYERS = 4

# Training images. It's a numpy array of size 60000x784.
training = load_images("dataset/train-images-idx3-ubyte/train-images-idx3-ubyte")
labels = load_labels("dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte")

# The derivative of ReLU. If x > 0, it computes
# the derivative of x, which is 1.0, otherwise
# ReLU returns a constant (0.0) and thus the 
# derivative is 0.
def drelu(x):
    return (x > 0.0).astype(mx.float32)

# Cross-entropy loss function. Given y (the
# actual value) and y_hat (the predicted value)
# the function returns a scalar telling how
# wrong the prediction is. This function is
# very convenient since the derivative of it
# when using softmax to distribute the predictions
# is just y_hat - y.
def cross_entropy(y, y_hat):
    y_hat = mx.clip(y_hat, 1e-9, 1 - 1e-9)
    return -mx.mean(mx.sum(y * mx.log(y_hat), axis=1))

# Input layer weights and biases
W1 = mx.random.normal((28*28, HIDDEN_SIZE)) * math.sqrt(2 / (28*28))
b1 = mx.zeros((HIDDEN_SIZE,))

# Output layer weights and biases
W_out = mx.random.normal((HIDDEN_SIZE, 10)) * math.sqrt(2 / HIDDEN_SIZE)
b_out = mx.zeros((10,))

# Hidden layers weights and biases
W_hidden = [mx.random.normal((HIDDEN_SIZE, HIDDEN_SIZE)) * math.sqrt(2 / HIDDEN_SIZE) for _ in range(HIDDEN_LAYERS)]
b_hidden = [mx.zeros((HIDDEN_SIZE,)) for _ in range(HIDDEN_LAYERS)]

# These are the matrices needed for Adam optimization.
# for specifics on the implementation of the algorithm
# refer to https://arxiv.org/abs/1412.6980

# Input weights moment
mW1 = mx.zeros_like(W1)
vW1 = mx.zeros_like(W1)
# Input biases moment
mb1 = mx.zeros_like(b1)
vb1 = mx.zeros_like(b1)
# Hidden layers weights moment
mW_hidden = [mx.zeros_like(W_hidden[0]) for _ in range(HIDDEN_LAYERS)]
vW_hidden = [mx.zeros_like(W_hidden[0]) for _ in range(HIDDEN_LAYERS)]
# Hidden layers biases moment
mb_hidden = [mx.zeros_like(b_hidden[0]) for _ in range(HIDDEN_LAYERS)]
vb_hidden = [mx.zeros_like(b_hidden[0]) for _ in range(HIDDEN_LAYERS)]
# Output weights moment
mW_out = mx.zeros_like(W_out)
vW_out = mx.zeros_like(W_out)
# Output biases moment
mb_out = mx.zeros_like(b_out)
vb_out = mx.zeros_like(b_out)

t = 0

total_steps = EPOCHS * (training.shape[0] / BATCH_SIZE)
steps = 0

# Training pass, repeat the training process
# for an EPOCHS amount of iterations.
for epoch in range(EPOCHS):
    for i in range(0, training.shape[0], BATCH_SIZE):
        label_batch = mx.zeros((BATCH_SIZE, 10))
        batch = mx.array(training[i:i+BATCH_SIZE])
        for j, lbl in enumerate(labels[i:i+BATCH_SIZE]):
            label_batch[j, int(lbl)] = 1.0
        # Forward pass. The forward pass is what does the actual prediction, using
        # the current parameters, for a given input batch X

        # Input layer regression
        z1 = linear(batch, W1, b1)
        # Apply activation
        a1 = relu(z1)

        # Hidden layers regression and activation
        z_hidden = [linear(a1, W_hidden[0], b_hidden[0])]
        a_hidden = [relu(z_hidden[0])]
        for l in range(1, HIDDEN_LAYERS):
            z_hidden.append(linear(a_hidden[-1], W_hidden[l], b_hidden[l]))
            a_hidden.append(relu(z_hidden[l]))


        # Output layer regression
        z_out = linear(a_hidden[-1], W_out, b_out)
        # Apply softmax
        y_hat = softmax(z_out)

        # Back propagation. This is where the model "learns"
        # by computing how wrong it is to then proceed to
        # adjust parameters.

        # Calculate the loss using cross-entropy
        loss = cross_entropy(label_batch, y_hat)

        # Calculate the gradient of the loss function w.r.t.
        # the output.
        d_loss = (y_hat - label_batch) / batch.shape[0]
        # Calculate the gradient of the loss function w.r.t.
        # the weights of the output layer.
        dW_out = a_hidden[-1].T @ d_loss
        # Calculate the gradient of the loss function w.r.t.
        # the biases of the output layer.
        db_out = mx.sum(d_loss, axis=0)

        # Calculate the gradient of the loss function w.r.t.
        # the weights and biases of all the hidden layers.


        # Gradients of hidden layers
        dW_hidden = [None] * HIDDEN_LAYERS
        db_hidden = [None] * HIDDEN_LAYERS
        # Gradient of previous layer
        d_hidden = d_loss
        for l in reversed(range(HIDDEN_LAYERS)):
            WT = W_out.T if l == HIDDEN_LAYERS - 1 else W_hidden[l + 1].T
            d_hidden = (d_hidden @ WT) * drelu(z_hidden[l])
            a_prev = a1 if l == 0 else a_hidden[l - 1]
            dW_hidden[l] = a_prev.T @ d_hidden
            db_hidden[l] = mx.sum(d_hidden, axis=0)

        # Backpropagate through the first hidden layer
        d1 = (d_hidden @ W_hidden[0].T) * drelu(z_hidden[0])
        # Compute gradients for the input layer weights and biases
        dW1 = batch.T @ d1
        # Calculate the gradient of the loss function w.r.t.
        # the biases of the input layer.
        db1 = mx.sum(d1, axis=0)

        # Adam optimization.
        
        t += 1

        # Input layer weights optimization with bias correction.
        for theta, gradient, m, v in zip(
            [W1, *W_hidden, W_out],
            [dW1, *dW_hidden, dW_out],
            [mW1, *mW_hidden, mW_out],
            [vW1, *vW_hidden, vW_out],
        ):
            m[:] = BETA1 * m + (1 - BETA1) * gradient
            v[:] = BETA2 * v + (1 - BETA2) * (gradient ** 2)
            m_hat = m / (1 - (BETA1 ** t))
            v_hat = v / (1 - (BETA2 ** t))
            theta[:] -= LEARNING_RATE * m_hat / (mx.sqrt(v_hat) + EPSILON)

        # Input layer biases optimization.
        for theta, gradient, m, v in zip(
            [b1, *b_hidden, b_out],
            [db1, *db_hidden, db_out],
            [mb1, *mb_hidden, mb_out],
            [vb1, *vb_hidden, vb_out],
        ):
            m[:] = BETA1 * m + (1 - BETA1) * gradient
            v[:] = BETA2 * v + (1 - BETA2) * (gradient ** 2)
            m_hat = m / (1 - (BETA1 ** t))
            v_hat = v / (1 - (BETA2 ** t))
            theta[:] -= LEARNING_RATE * m_hat / (mx.sqrt(v_hat) + EPSILON)
        steps += 1
        if steps % 100 == 0:
            mx.eval(m)
            mx.eval(v)
        if steps % 500 == 0:
            print(f"\rtraining: {(100 * (steps / total_steps)):.2f}%, loss: {loss}", end="")

    LEARNING_RATE *= 0.99

print("\nTraining complete.")
np.savetxt("weights/weights1.txt", np.array(W1))
np.savetxt("biases/biases1.txt", np.array(b1))
for i in range(HIDDEN_LAYERS):
    np.savetxt(f"weights/weights_hidden{i}.txt", np.array(W_hidden[i]))
    np.savetxt(f"biases/biases_hidden{i}.txt", np.array(b_hidden[i]))
np.savetxt("weights/weights3.txt", np.array(W_out))
np.savetxt("biases/biases3.txt", np.array(b_out))
