import os
import cv2
from keras import optimizers
from keras.utils import to_categorical

# TODO: ask if we can add conf here
from conf import Conf
c = Conf()
from preprocessor import *

def get_t(images):
    n_images = len(images)
    if n_images <= 2**2+2:
        return 2
    if n_images <= 4**2+4:
        return 4
    return 5


def get_image_size(t):
    if t == 2 or t == 4:
        return 32
    return 16


def is_image(images):
    votes_false = 0
    first = True
    images = np.array(images) / 255.
    for image in images:
        hist = np.histogram(image, bins=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])[0]
        # else:
        #     hist_accumulated += np.histogram(image, bins=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])[0]
        if hist[-1] > hist[0]:  # more white than black
            votes_false += 1
        print(hist)
    print(votes_false)

    if votes_false >= len(images) // 2:  # majority vote
        return False
    return True


def read_test_images_docs(file_dir):
    files = os.listdir(file_dir)
    # files.remove('.DS_Store')
    files.sort()
    images = []
    X_batch = []
    y_batch = []
    labels_in_folder = []
    for f in files:
        print(f)
        if f == '.DS_Store':
            continue
        im = cv2.imread(file_dir + f)
        try:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        except:
            continue
        images.append(im)
    return images
    print(is_image(images))



# def predict(images):
def predict(images, y_batch):
    labels = []
    X_batch = []
    t = get_t(images)
    img_size_dst = get_image_size(t)
    c = Conf(int(t), int(img_size_dst), is_image(images))

    # adam = optimizers.Adam()

    resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])

    # resnet.load_weights("test.h5")
    resnet.load_weights(
        "resnet_maxSize_32_tilesPerDim_2_nTilesPerSample_6_isImg_True_mID_0_1548231669.6831458_L_0.8997396.h5")

    resized_images = []
    original_images = []
    for im in images:
        original_images.append(im)
        resized_images.append(resize_image(im, c.max_size, simple_reshape=True))  # TODO: has to be here! Only after determining doc / image

    resized_images = add_similarity_channel(resized_images, original_images, c)

    X_batch.append(np.array(resized_images))  # a folder is one single sample
    # print(labels_in_folder)
    # folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    # y_batch.append(folder_labels)
    # if len(y_batch) == batch_size:
        # print(np.array(X_batch).ndim)
        # print(np.array(X_batch))
        # if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size, 2):
        #     print(folder)
        #     print(np.array(X_batch).shape)
    X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size, 2))

    # X_batch.append(np.array(add_similarity_channel(resized_images, original_images, c)))
    # # print(X_batch.shape)
    # X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size, 2))
    print(resnet.evaluate(X_batch, y_batch))
    # here comes your code to predict the labels of the images
    return labels


from resnet_adapted import *


# def calc_edge_similarity_score(images):
#     from scipy import spatial  # TODO: add to dependencies
#     sim_layer = np.zeros((c.max_size, c.max_size))
#     for i in range(len(images)):
#         n_columns = 1
#         im = images[i] / 255.
#         print(im.shape)
#         right_edge_original = im[:, -n_columns:].flatten()
#         left_edge_original = im[:, 0:n_columns].flatten()
#         top_edge_original = im[0, :].flatten()
#         bottom_edge_original = im[-1, :].flatten()
#         row = 0
#         for j in range(len(images)):
#             if i == j:
#                 continue
#             potential_neighbor = images[j] / 255.
#             right_edge = [potential_neighbor[:,0:1].flatten()]
#             left_edge = [potential_neighbor[:,-1:].flatten()]
#             top_edge = potential_neighbor[0, :].flatten()
#             bottom_edge = potential_neighbor[-1, :].flatten()
#             # print(sum(right_edge_original - left_edge))
#             cosine_sim_rl = 1-spatial.distance.cosine(right_edge_original, left_edge)
#             cosine_sim_lr = 1 - spatial.distance.cosine(left_edge_original, right_edge)
#             cosine_sim_tb = 1 - spatial.distance.cosine(top_edge_original, bottom_edge)
#             cosine_sim_bt = 1 - spatial.distance.cosine(bottom_edge_original, top_edge)
#             sim_layer[row,0:4] = cosine_sim_lr, cosine_sim_rl, cosine_sim_tb, cosine_sim_bt
#             row += 1
#         final_image = np.zeros((32, 32, 2))
#         final_image[:,:,0] = images[i]
#         final_image[:,:,1] = sim_layer
#         images[i] = final_image  # has two channels, the second channel has the similarities
#     return images


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

        images.append(im)
        label = f.split('_')[-1].split('.')[0]
        if label == "-1":
            # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
            label = c.n_original_tiles
        labels_in_folder.append(label)

    # counter = 0
    # for f in files:#range(c.tiles_per_dim):  # TODO - fix to match up to t OoD in folder
    #     im = np.zeros((c.max_size, c.max_size))
    #     # label = "-1"
    #     # if f == '.DS_Store':
    #     #     continue
    #     if counter == c.tiles_per_dim:
    #         break
    #     im = cv2.imread(file_dir + f)
    #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #
    #     images.append(im)
    #
    #     # label = f.split('_')[-1].split('.')[0]
    #     if label == "-1":
    #         # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
    #         label = c.n_original_tiles
    #     # labels_in_folder.append(label)
    #
    #     counter += 1

    folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    y_batch.append(folder_labels)
    y_batch = list(np.array(y_batch).reshape(c.n_tiles_per_sample, 1, c.n_classes))

    Y = predict(images, y_batch) # TODO
    # Y = predict(images) # TODO
    print(Y)
    return Y


evaluate('dataset_2_isImg_True/n01440764_172_crw_0_crh_0_reshape_False/')
# read_test_images_docs('dataset_5_isImg_False/73_5_crw_0_crh_15_reshape_True/')
# read_test_images_docs('dataset_5_isImg_True/n01440764_7267_crw_0_crh_45_reshape_False/')
# calc_edge_similarity_score(read_test_images_docs('example/'))