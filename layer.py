import numpy as np

class denseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.output_size, self.input_size)
        self.biases = np.random.randn(output_size, 1)
    def forward(self, input):
        self.input = input
        result = np.dot(self.weights, input) + self.biases
        return result
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, np.transpose(self.input))
        input_gradient = np.dot(np.transpose(self.weights), output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class sigmoid:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, input):
        self.input = input
        return self.sigmoid(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.sigmoid_prime(self.input))
