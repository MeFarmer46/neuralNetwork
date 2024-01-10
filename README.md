# Python neural network

Python code to train your own neural network, configured for the MNIST dataset.

<br>

## How to use
By default, the networks are configured to train and predict on the MNIST dataset.
<br>
The network settings can be changed by changing the variables in the mnist.py file. Run this file to use the network or modify it for other purposes and datasets.
<br>
Comment out the training part in mnist.py and use the predict function te use the network for predicting.
<br>

Data <br>
Training data consists of two arrays. One with 28x28 images and another array with the labels. These are preprocessed to feed through the network. After the preprocess function there are, once again, two arrays. One is a list of lists where each list has 784 values between 0 and 1 for the individual pixels. The other is a list of lists where each list has 10 values where only one of the is 1 and the others are zero. This number corresponds with the label.

<br>

## Available networks
Pre-trained networks for the MNIST dataset. Can be used by changing the network_path variable.

#### Network 1
Short network with 3 layers and few neurons. Trained to about 95% accuracy on test examples. Training the network more is unlikely to yield better results as the network is very small and overfitting may occur.
<br> <br>
Network can be used by setting the network_path variable to "./networks/net1"
<br> <br>
Network variable: <br>
network = [ <br>
    denseLayer(784, 40), <br>
    sigmoid(), <br>
    denseLayer(40, 20), <br>
    sigmoid(), <br>
    denseLayer(20, 10), <br>
    sigmoid() <br>
] <br>

<br>

#### Network 2
Long network with 5 layers and a lot of neurons. Trained to about 96% accuracy on test examples. Training the network more will lead to overfitting and poorer results.
<br> <br>
Network can be used by setting the network_path variable to "./networks/net2"
<br> <br>
Network variable: <br>
network = [ <br>
    denseLayer(784, 100),  <br>
    sigmoid(),  <br>
    denseLayer(100, 100),  <br>
    sigmoid(),  <br>
    denseLayer(100, 100),  <br>
    sigmoid(),  <br>
    denseLayer(100, 40),  <br>
    sigmoid(),  <br>
    denseLayer(40, 10), <br>
    sigmoid() <br>
] <br>