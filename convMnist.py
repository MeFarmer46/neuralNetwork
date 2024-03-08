import cv2
import numpy as np
from keras.datasets import mnist as mnist_imp

def convert(index):
    (train_images, train_labels), (test_images, test_labels) = mnist_imp.load_data()

    for i in index:
        digit = test_images[i]
        digit_normalized = digit / 255.0
        digit_uint8 = (digit_normalized * 255).astype(np.uint8)
        digit_uint8 = cv2.bitwise_not(digit_uint8)
    
        label = test_labels[i]

        img_path = f"wrongImages/({i}) - {label}.png"
        cv2.imwrite(img_path, digit_uint8)

