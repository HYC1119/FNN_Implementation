# FNN_Implementation from Scratch
Implement a feedforward neural network (FNN) model to classify the MNIST dataset.

Design a FNN model architecture with 2 hidden layers and perform the random initialization for model weights. 
The first hidden layer contains 100 neurons and the second one contains 150 neurons. 
Run backpropagation algorithm and use mini-batch SGD (stochastic gradient descent) to optimize the parameters. Use cross-entropy loss as loss function.

## Dataset
[MNIST dataset source](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

Data in `mnist.npz`  
|Name|Dimension|Description|
|--|--|--|
|x_train|(60000,28,28,1)|60000 training images with shape (28,28,1)|
|y_train|(60000,)|corresponding labels of training images|
|x_test|(10000,28,28,1)|10000 testing images with shape (28,28,1)|
|y_test|(10000,)|corresponding labels of testing images|
