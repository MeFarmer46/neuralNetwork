from keras.datasets import mnist as mnist_imp
import keras.utils

class data:
    def preprocess(self, x, y, limit):
        # reshape and normalize input data
        x = x.reshape(x.shape[0], 28 * 28, 1)
        x = x.astype("float32") / 255
        # encode output which is a number in range [0,9] into a vector of size 10
        y = keras.utils.to_categorical(y)
        y = y.reshape(y.shape[0], 10, 1)
        return x[:limit], y[:limit]

    def mnist(self, amount_train=60000, amount_test=10000):
        (train_images, train_labels), (test_images, test_labels) = mnist_imp.load_data()
        x_train, y_train = self.preprocess(train_images, train_labels, amount_train)
        x_test, y_test = self.preprocess(test_images, test_labels, amount_test)
        return x_train, y_train, x_test, y_test
        