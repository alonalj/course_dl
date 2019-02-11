'''

We assume that all input texts/images are given in gray-scale. Thus, the
given shredder is transforming all dataset to gray-scale. Make sure you also do so.

You should create an algorithm that gets as input a folder containing jpeg
images of a shredded document (a picture or an image of a scanned document)
and recovers it to the original text/image. For evaluation, your algorithm should
return a label for each crop. The label should be a number in {0, 1, 2, 3, ...t2−1},
representing the crop’s position in the original document, assuming that the
crops are arranged in a “row-major order” (see Figure 1). For evaluation we will
use the supplied script “evaluate.py”. You should complete this file with your
own prediction method. Your algorithm should be able to recover documents
and images of various sizes. You will be evaluated on t values of 2, 4, and 5.

Additionally, the input folder with the jpeg images will contain up to t outof-distribution (OoD) images. That is, there will be t
2 pieces that originally
came from the same image or scanned-document, and additional up to t pieces
which do not belong to the main image. Your algorithm should distinguish these
OoD pieces, and label each of them as −1 (minus one). You can assume that
pictures will contain picture-OoD distractors, and that documents will contain
document-OoD distractors.

The type of the input images (i.e., pictures or scanned documents) is not
given. Your algorithm should discover it on its own for each new set of
pieces

'''

'''
PLAN
Repeat the following for docs too

# Data prep
dataset augmentation - this has to be before tiling otherwise tiles might not match: 
(1) crop images (into smaller crops only to not include hints (e.g. black edges) of which tile this is     TODO
(2) all other relevant augmentations we saw in the code provided for transfer learning in ex. 3            TODO
for image,
    for augmentation of image:
        shred in preprocessor into subfolders with labels in name ( check keras input?)
for each folder randomly select t ood tiles from another folder and add them 
split folders into train, val, test parent-folders randomly, using dictionary

# Model
Repeat for docs:
Resnet 

input: all tiles in folder (batch = multiple folders, one sample = one folder)
intermediate output : one hot vector per tile in sample (folder), simultaneously predicted
regularization: one hots have to be orthogonal (force each tile to have a different position) -> justify in report with
helping better define hypothesis space
predict: labels

# create eval function in eval.py

# Baseline model? Something naive?
'''
from resnet_ood_pairs import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
import keras.backend as K
from preprocessor import *  # TODO verify all these imports work for terminal + pycharm proj. opened from scratch
import cv2
from conf import Conf
from saving_m import save_model


def lr_scheduler(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.1
    if epoch > 100:
        lr = 0.01
    if epoch > 150:
        lr = 0.001
    if epoch > 200:
        lr = 1e-4
    return lr


def get_gauss_noise(image):
    row, col = image.shape
    mean = 0
    var = 0.0004
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    return gauss


def noisy(noise_typ, image, gauss=None):
    if noise_typ == "gauss":
        row, col = image.shape
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.006
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out


def scale_img(image):
    image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)


def data_generator(data_type, tiles_per_dim, data_split_dict, batch_size, c):
    import random
    import glob


    path = "ood_isImg_{}".format(c.is_images)
    if data_type == "val":
        path = path+'_val'
    if data_type == "test":
        path = path + '_test'
    folders_two_class = os.listdir(path)
    train_len = len(glob.glob(path+'/0'+'/*'))
    for i in range(train_len // batch_size):
        X_batch = []
        y_batch = []
        for class_folder in folders_two_class:
            label = class_folder
            original_images = []
            folders_in_class = glob.glob(path+'/'+class_folder+'/*')
            np.random.shuffle(folders_in_class)  # random shuffle files in folders too  #TODO: evaluate uses sorted files... is this necessary?
            for folder in folders_in_class[:batch_size//2]: # because of random shuffle above, will be different between yields
                combined_images = []
                labels = []
                labels.append(label)
                files_in_folder = glob.glob(folder+'/*')
                combined_images = []
                for f in files_in_folder:
                    im = cv2.imread(f)
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    im = resize_image(im, max_size=c.max_size, simple_reshape=True)
                    combined_images.append(im / 255.)
                combined_images = np.concatenate(combined_images, axis=1)
                combined_images.resize(c.max_size, c.max_size)
                combined_images = [combined_images]
                X_batch.append(np.array(combined_images))  # a folder is one single sample
                # print(labels_in_folder)
                folder_labels = to_categorical(labels, num_classes=2)
                y_batch.append(folder_labels)

                # if noise:
                #     gauss = get_gauss_noise(im_resized)
                #     im_resized = noisy("gauss",im_resized, gauss)

                # if im_resized.shape != (c.max_size, c.max_size):
                #     print("Bad shape for folder {}, file {}".format(folder, f))

                # if flip_img:
                #     im_resized = cv2.flip(im_resized, 1)

        if len(y_batch) == batch_size:
            zipped = list(zip(X_batch, y_batch))
            random.shuffle(zipped)
            X_batch, y_batch = zip(*zipped)

            # if np.array(X_batch).shape[1:] != (2, c.max_size, c.max_size, 2):
            #     print(folder)
            #     print(np.array(X_batch).shape)
            # print(list(np.array(y_batch).reshape(1, batch_size, 2)))
            combined_images, labels = list(np.array(X_batch).reshape(1, batch_size, c.max_size, c.max_size, 1)), list(np.array(y_batch).reshape(1, batch_size, 2))
            # print(labels)
            yield combined_images, labels


def run(c):

    batch_size = 128
    c.max_size = 64
    # adam = optimizers.Adam()
    if c.n_tiles_per_sample > 6:
        batch_size = 50
    if c.n_tiles_per_sample > 20:
        batch_size = 32
        # adam = optimizers.Adam(0.0001)

    maxepoches = 800
    # learning_rate = 0.1

    # train_generator = data_generator("train", c.tiles_per_dim, c.data_split_dict, batch_size)
    # val_generator = data_generator("val", c.tiles_per_dim, c.data_split_dict, batch_size)
    # n_samples_train = len(load_obj( c.data_split_dict)["train"])

    # # TODO: NOTE: batch size DOESN'T always have to be divisible by t^2 (data_generator only yields after processing a WHOLE folder, so keras will treat as a single sample )
    # for i in range(2):
    #     print(dgen.__next__()[1])

    resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    # sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",  # switch to adam later
        metrics=['accuracy']
    )
    resnet.summary()
    # save_model(resnet, "test_m.h5")

    no_improvement_tolerance = 10000
    no_improvement_counter = 0
    val_steps_max = 0
    best_avg_acc_val = 0
    best_total_loss = np.inf

    for e in range(maxepoches):
        print("Epoch {}".format(e))
        train_generator = data_generator("train", c.tiles_per_dim, c.data_split_dict, batch_size, c)
        step = 0
        for X_batch, y_batch in train_generator:
            # print(X_batch.shape)
            # print(y_batch.shape)
            hist = resnet.train_on_batch(X_batch, y_batch)  # , batch_size, epochs=maxepoches)
            print("TRAIN STATS:", hist)
            preds = resnet.predict_on_batch(X_batch)

            if step % 5 == 0:
                print("hist", hist)
            if step % 100 == 0:
                step += 1
                preds = np.array(preds)
                y = np.array(y_batch)
                # print(preds)
                # print(preds - y)
                # assert preds.shape == y.shape


                # Validating at end of epoch
                print("VALIDATING")
                val_generator = data_generator("val", c.tiles_per_dim, c.data_split_dict, batch_size, c)
                current_acc = []
                for X_batch_val, y_batch_val in val_generator:
                    hist_val = resnet.test_on_batch(X_batch_val, y_batch_val)
                    current_acc.append(hist_val[-1])
                current_avg_acc = np.mean(current_acc)
                if current_avg_acc > best_avg_acc_val:
                    resnet.save_weights(
                        'ood_resnet_maxSize_isImg_{}_L_{}.h5'.format(c.max_size,   current_avg_acc))


                    print("val hist", hist_val)
                    best_avg_acc_val = current_avg_acc
                    print("best avg acc val: {}".format(best_avg_acc_val))
                    no_improvement_counter = 0  # reset
                else:
                    no_improvement_counter += 1
                # # saving train ckpt
                # resnet.save_weights(
                #     'train_resnet_maxSize_{}_tilesPerDim_{}_nTilesPerSample_{}_isImg_{}_mID_{}_L_{}.h5'.format(c.max_size,
                #                                                                                                c.tiles_per_dim,
                #                                                                                                c.n_tiles_per_sample,
                #                                                                                                c.is_images,
                #                                                                                                c.mID,
                #                                                                                                str(
                #                                                                                                    hist[0])))

                print("acc", current_avg_acc)
                val_steps_max += 1

                if no_improvement_counter >= no_improvement_tolerance:
                    print("No improvement for {} validation steps. Stopping.".format(no_improvement_tolerance))
                    return

                # if val_steps_max == 5:
                #     print("Finished validating on {} batches".format(val_steps_max))
                #     break
                # callbacks=[reduce_lr])

            # resnet_cifar_10_history = resnet.fit_generator(train_generator,
            #                                                steps_per_epoch=n_samples_train // batch_size,
            #                                                epochs=maxepoches,
            #                                                validation_data=val_generator, validation_steps=1)#,
            #                                                # callbacks=[reduce_lr])


if __name__ == '__main__':
    c = Conf()
    c.is_images = True
    run(c)
