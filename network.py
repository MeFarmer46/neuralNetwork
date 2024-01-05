import time
import numpy as np

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(save, network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        start_time = time.time()
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        end_time = time.time()
        epoch_time = round(end_time - start_time)
        if verbose:
            print(f"{e + 1}/{epochs}, time: {epoch_time}, error={round(error, 5)}")
        save.write(network)

def test(network, x, y, type="test set"):
    def read_output(output):
        return np.argmax(output)

    right = 0
    wrong = 0
    total = 0

    z = 0
    for i in x:
        output = read_output(predict(network, i))
        label = np.argmax(y[z])
        z += 1
        total += 1
        if output == label:
            right += 1
        else:
            wrong += 1

    print(f"Network testing complete. Tested set: {type}")
    print(f"Success rate: {round(right / total * 100, 1)}% \n")