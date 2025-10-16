# mnist
This is a neural network trained to predict images of the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database).
It's written in python without making use of ML libraries directly, using [Apple's MLX](https://github.com/ml-explore/mlx) to accelerate matrix computations with the GPU.
This makes the scripts very fast on Apple Silicon machines since MLX makes great use of the SoC shared memory and lazy evaluation for efficiency.
The code is commented in detail and quite simple to understand with basic understanding of neural networks.

## Implementation
The network is structured to use 4 hidden layers of size 256x256. For activation of hidden layers [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) is used
and loss is calculated with [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy). For optimization Adam is used (check the references section).

## Running
To run the scripts you first need to download the dataset and the dependencies with
```sh
./data.sh
pip install -r requirements.txt
```
After that you can train the model by running the `train.py` script. To test the model you can run `test.py` instead. Accuracy should be around 97%.

## References
- [Adam optimization algorithm](https://arxiv.org/abs/1412.6980)
- [Google ML crash course](https://developers.google.com/machine-learning/crash-course)
