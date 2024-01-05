from loss import mse_prime, mse
from network import train, predict, test
from file import objectSave
from layer import denseLayer, sigmoid
from data import data

import time

data_var = data()

# Network settings
network_path = "./networks/net3"    # defines the path where te network will be safed

example_amount_train = 60000        # amount of training examples used. Max 60,000
example_amount_test = 10000         # amount of test examples used. Max 10,000

learning_rate = 0.1                # network learning rate
amount_epochs = 50                 # amount of epochs used


network_save = objectSave(network_path)

x_train, y_train, x_test, y_test = data_var.mnist(amount_train=example_amount_train, amount_test=example_amount_test)

# Create new network
# network = [
#     denseLayer(784, 40),
#     sigmoid(),
#     denseLayer(40, 20),
#     sigmoid(),
#     denseLayer(20, 10),
#     sigmoid()
# ]

# Or open existing one
network = network_save.open()

# Train the network with selected parameters and measure the time
start_time = time.time()
train(network_save, network, mse, mse_prime, x_train, y_train, epochs=amount_epochs, learning_rate=learning_rate)
end_time = time.time()
print(f"\n\nTotal training time: {round((end_time-start_time)/60)} minutes")

# Test the network on both sets
test(network, x_test, y_test, "test set")
test(network, x_train, y_train, "training set")