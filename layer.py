import numpy as np

class denseLayer:
    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.weights = np.random.randn(self.outputSize, self.inputSize) # Generate a list of lists with random weights
        self.biases = np.random.randn(outputSize, 1) # Generate a list of biases for each neuron
    def forward(self, input):
        self.input = input
        result = np.dot(self.weights, input) + self.biases
        return result
    def backward(self, outputGradient, learningRate):
        weightsGradient = np.dot(outputGradient, np.transpose(self.input))
        inputGradient = np.dot(np.transpose(self.weights), outputGradient)
        self.weights -= learningRate * weightsGradient
        self.biases -= learningRate * outputGradient
        return inputGradient

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
