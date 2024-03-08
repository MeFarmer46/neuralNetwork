import cv2
import numpy as np
from network import predict

# Predict for png's in the path folder and print the results.
def testReal(network, path='testImages/', amount=10):
    testImages = []
    
    i = 0
    while i < amount:
        img_path = f"{path}{i}.png"
        img = cv2.bitwise_not(cv2.imread(img_path, 0)) / 225
        endArray = []
        for x in img:
            for y in x:
                endArray.append([y])

        testImages.append(endArray)
        i += 1

    for current in testImages:
        print(np.argmax(predict(network, current)))