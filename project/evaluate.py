import os
import cv2
from keras.utils import to_categorical
from preprocessor import *
from resnet_adapted import *
from conf import Conf
from resnet_img_doc_classifier import *
c = Conf()


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
    images = np.array(images)
    resnet_img_vs_doc = build_resnet_img_vs_doc(1e-3)
    resnet_img_vs_doc.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    resnet_img_vs_doc.load_weights('is_img_or_doc.h5')
    resized_images = []
    for im in images:
        resized_images.append(cv2.resize(im, (32, 32)).reshape((32, 32, 1)))#.reshape((32,32,1)))
    # resnet_img_vs_doc.predict_on_batch(np.array([cv2.resize(im, (32, 32)).reshape((32, 32, 1))]))
    resized_images = np.array(resized_images)
    preds = resnet_img_vs_doc.predict_on_batch(resized_images)
    preds = preds.argmax(axis=1)
    if np.median(preds) == 1:
        return True
    return False


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
    isImg = is_image(images)

    resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])

    if isImg:
        resnet.load_weights(
            "resnet_maxSize_32_tilesPerDim_2_nTilesPerSample_6_isImg_True_mID_0_1548231669.6831458_L_0.93424475.h5")
    else:
        print("FAILED")
        resnet.load_weights(
            "resnet_maxSize_32_tilesPerDim_2_nTilesPerSample_6_isImg_True_mID_0_1548231669.6831458_L_0.8997396.h5")

    resized_images = []
    original_images = []
    im_shapes = {}
    add_n_null_images = 0
    if len(images) < c.n_tiles_per_sample:
        add_n_null_images = c.n_tiles_per_sample - len(images)
    for im in images:
        original_images.append(im)
        if im.shape not in im_shapes:
            im_shapes[im.shape] = 1
        else:
            im_shapes[im.shape] += 1
        resized_images.append(resize_image(im, c.max_size, simple_reshape=True))  # TODO: has to be here! Only after determining doc / image

    # adding additional OoD images to reach t OoDs per sampled folder
    for n_null in range(add_n_null_images):
        resized_images.append(np.zeros((c.max_size, c.max_size)))
        original_images.append(np.zeros((c.max_size, c.max_size)))
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
    print(resnet.predict_on_batch(X_batch))
    logits = resnet.predict_on_batch(X_batch)
    for l in logits:
        idx_max = l.argmax(axis=1)
        idx_max = int(idx_max)
        if idx_max == c.n_classes - 1:
            # OoD
            idx_max = -1
        labels.append(idx_max)
    print("before ood", labels)
    # in case OoDs are of a different shape, we can easily label them:
    for im_idx in range(len(images)):
        im = images[im_idx]
        if im_shapes[im.shape] <= t:
            labels[im_idx] = -1
    print("after ood", labels)
    # here comes your code to predict the labels of the images
    return labels


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
        label = int(f.split('_')[-1].split('.')[0])

        if label == "-1":
            # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
            label = c.n_original_tiles
        labels_in_folder.append(label)

    print(labels_in_folder)
    # in case we're adding OoD null image
    if len(labels_in_folder) < c.n_tiles_per_sample:
        for k in range(c.n_tiles_per_sample-len(labels_in_folder)):
            labels_in_folder.append(-1)

    folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    y_batch.append(folder_labels)
    y_batch = list(np.array(y_batch).reshape(c.n_tiles_per_sample, 1, c.n_classes))

    Y = predict(images, y_batch) # TODO
    # Y = predict(images) # TODO
    print(Y)
    return Y

try:
    files_dict = load_obj('train_test_val_dict_img_2')
    files = files_dict['test']
    for f in files:
        # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
        evaluate('dataset_2_isImg_True/'+f+'/')
except:
    files_dict = load_obj('train_test_val_dict_doc_2')
    files = files_dict['test']
    for f in files:
        # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
        evaluate('dataset_2_isImg_False/' + f + '/')
    # filename = f.split('.')[0] + '_crw_' + str(c_w) + '_crh_' + str(c_h) + '_reshape_' + str(reshape)
# evaluate('dataset_2_isImg_True/n01440764_18_crw_0_crh_0_reshape_False/')
# read_test_images_docs('dataset_5_isImg_False/73_5_crw_0_crh_15_reshape_True/')
# read_test_images_docs('dataset_5_isImg_True/n01440764_7267_crw_0_crh_45_reshape_False/')
# calc_edge_similarity_score(read_test_images_docs('example/'))
