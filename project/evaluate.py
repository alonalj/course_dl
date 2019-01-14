import os
import cv2
from preprocessor import resize_image

# TODO: ask if we can add conf here
from conf import Conf
c = Conf()

def predict(images):
    labels = []
    # TODO: Use ensmble for each one

    # here comes your code to predict the labels of the images
    return labels



def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = resize_image(im, c.max_size)  # TODO: ask if we can add functions here
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)
        # TODO: create phantom image of 0s OoD if number of files is not


    Y = predict(images)
    return Y


