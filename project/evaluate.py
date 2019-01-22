import os
import cv2
from keras import optimizers
from keras.utils import to_categorical

from preprocessor import resize_image

# TODO: ask if we can add conf here
from conf import Conf
c = Conf()

def predict(images):
    labels = []
    # TODO: Use ensmble for each one

    # here comes your code to predict the labels of the images
    return labels
from resnet_adapted import *

c = Conf(int(5), int(16), True, 0, "")
adam = optimizers.Adam()

resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)


resnet.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy'])

# resnet.load_weights("test.h5")

from keras.models import load_model
resnet = load_model("TESTING.h5")
# resnet.load_weights("resnet_maxSize_32_tilesPerDim_4_nTilesPerSample_20_isImg_True_mID_0_1547976802.593452_L_120.22917.h5")


def evaluate(file_dir='output/'):
    files = os.listdir(file_dir)
    # files.remove('.DS_Store')
    files.sort()
    images = []
    X_batch = []
    y_batch = []
    labels_in_folder = []
    for f in files:
        if f == '.DS_Store':
            continue
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = resize_image(im, c.max_size)

        images.append(im)
        label = f.split('_')[-1].split('.')[0]
        if label == "-1":
            # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
            label = c.n_original_tiles
        labels_in_folder.append(label)

    counter = 0
    for f in files:#range(c.tiles_per_dim):  # TODO - fix to match up to t OoD in folder
        # im = np.zeros((c.max_size, c.max_size))
        if f == '.DS_Store':
            continue
        if counter == c.tiles_per_dim:
            break
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = resize_image(im, c.max_size)

        images.append(im)

        label = f.split('_')[-1].split('.')[0]
        if label == "-1":
            # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
            label = c.n_original_tiles
        labels_in_folder.append(label)

        counter += 1

    folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    y_batch.append(folder_labels)
    y_batch = list(np.array(y_batch).reshape(c.n_tiles_per_sample, 1, c.n_classes))
    X_batch.append(np.array(images))
    X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size))
    Y = resnet.test_on_batch(X_batch, y_batch)


    # Y = predict(images) # TODO
    print(Y)
    return Y


evaluate()