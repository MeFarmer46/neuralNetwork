from loss import mse_prime, mse
from network import train, predict, test
from file import objectSave
from layer import denseLayer, sigmoid
from data import data
from convMnist import convert
from convpng import testReal

data_var = data()

# Network settings
network_path = "./networks/net3"    # defines the path where te network will be safed (net2 performs best)

example_amount_train = 60000        # amount of training examples used. Max 60,000
example_amount_test = 10000         # amount of test examples used. Max 10,000

learning_rate = 0.01                # network learning rate (0.01 standard)
amount_epochs = 50                 # amount of epochs used (At least 100 for a good network with a low learning rate)


network_save = objectSave(network_path)

x_train, y_train, x_test, y_test = data_var.mnist(amount_train=example_amount_train, amount_test=example_amount_test)

# Create new network
# network = [
#     denseLayer(784, 100),
#     sigmoid(),
#     denseLayer(100, 40),
#     sigmoid(),
#     denseLayer(40, 20),
#     sigmoid(),
#     denseLayer(20, 10),
#     sigmoid()
# ]

# Or open existing one
network = network_save.open()

# Train the network with selected parameters and measure the time
# train(network_save, network, mse, mse_prime, x_train, y_train, epochs=amount_epochs, learning_rate=learning_rate)


# Test the network on both sets
wrongNumbers = test(network, x_test, y_test, "test set", makeLog=True) # Store the wrong numbers from the test set as png's
test(network, x_train, y_train, "training set")

# Uncomment to convert the wrong guesses to png files and store them in the ./wrongImages folder
# convert(wrongNumbers)

# Test the network for the realistic test examples
testReal(network)