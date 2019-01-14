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
from resnet_adapted import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
import keras.backend as K
from preprocessor import *  # TODO verify all these imports work for terminal + pycharm proj. opened from scratch
import cv2
from conf import Conf


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
    if epoch>100:
        lr = 0.01
    if epoch>150:
        lr = 0.001
    if epoch>200:
        lr = 1e-4
    return lr


def data_generator(data_type, tiles_per_dim):
    import random
    split_dict = load_obj("train_test_val_dict")
    folders = split_dict[data_type]
    dataset_folder = "dataset_{}".format(tiles_per_dim)
    # while True:
    np.random.shuffle(folders)  # shuffle folders between epochs
    X_batch = []
    y_batch = []
    for folder in folders:
        skip_folder = False
        flip_img = False
        if random.random() > 0.5 and data_type == 'train':
            flip_img = True
        if c.only_images and len(folder) < 7:
            # print(len(folder))
            continue
        folder_path = dataset_folder+'/'+folder
        files = os.listdir(folder_path)
        np.random.shuffle(files)  # random shuffle files in folders too
        images_in_folder = []
        labels_in_folder = []
        # if len(files) != c.n_tiles_per_sample:
        #     print("Less than {} tiles in folder {}. Due to it being the first one of its type in preprocessor".format(c.n_tiles_per_sample, folder))
        for f in files:
            if skip_folder:
                continue
            label = f.split('_')[-1].split('.')[0]
            if label == "-1":
                # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
                label = c.n_original_tiles
            labels_in_folder.append(label)
            im = cv2.imread(folder_path + '/' + f)
            try:
                img_shape = im.shape[0]+im.shape[1]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            except:
                print("failed on {}".format(folder_path + '/' + f))
                skip_folder = True
                continue

            im_resized = resize_image(im, max_size=c.max_size)
            assert im_resized.shape == (c.max_size, c.max_size)
            if flip_img:
                im_resized = cv2.flip(im_resized, 1)
            images_in_folder.append(im_resized)

        if np.array(images_in_folder).shape != (c.n_tiles_per_sample, c.max_size, c.max_size):
            continue

        X_batch.append(np.array(images_in_folder))  # a folder is one single sample
        # print(labels_in_folder)
        folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
        y_batch.append(folder_labels)
        if len(y_batch) == batch_size:
            # print(np.array(X_batch).ndim)
            # print(np.array(X_batch))
            if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size):
                print(folder)
                print(np.array(X_batch).shape)
            # TODO: insert a 0s OoD one for each batch too
            yield list(np.array(X_batch).reshape(c.n_tiles_per_sample, batch_size, c.max_size, c.max_size)), \
                  list(np.array(y_batch).reshape(c.n_tiles_per_sample, batch_size, c.n_classes))
            X_batch = []
            y_batch = []
    # handle last batch in case n_folders not fully divisible by batch_size (has a remainder)
    if len(y_batch) != batch_size:  # if equal, already yielded above
        # print(np.array(X_batch).shape)
        # print(np.array(X_batch))
        if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size):
            print(folder)
            print(np.array(X_batch).shape)
        yield list(np.array(X_batch).reshape(c.n_tiles_per_sample, -1, c.max_size, c.max_size)),\
              list(np.array(y_batch).reshape(c.n_tiles_per_sample, -1, c.n_classes))


def temp():
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    resnet.summary()



    datagen = ImageDataGenerator()
        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    resnet_cifar_10_history = resnet.fit_generator(datagen.flow(X_train, y_train,
                                                                batch_size=batch_size),
                                                   steps_per_epoch=X_train.shape[0] // batch_size,
                                                   epochs=maxepoches,
                                                   validation_data=(X_test, y_test),
                                                   callbacks=[reduce_lr])



# def dice_coef(y_true, y_pred, smooth, thresh):
#     y_pred = y_pred > thresh
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#
# def my_loss(y_true, y_pred):
#     cross_entropy_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
#     # print(y_true.shape)
#     y_p = K.reshape(y_pred, shape=(-1, c.n_classes))
#     sum = 0
#     for t in range(c.n_tiles_per_sample):
#         sum += keras.layers.Lambda(lambda x: x[t, :, :])(y_p)
#     all_differet_loss = K.sum((1- sum)**2, 1)
#
#     return cross_entropy_loss #+ all_differet_loss



if __name__ == '__main__':
    c = Conf()
    batch_size = 128
    maxepoches = 250
    learning_rate = 0.1

    train_generator = data_generator("train", c.tiles_per_dim)
    val_generator = data_generator("val", c.tiles_per_dim)
    n_samples_train = len(load_obj("train_test_val_dict")["train"])

    # # TODO: NOTE: batch size DOESN'T always have to be divisible by t^2 (data_generator only yields after processing a WHOLE folder, so keras will treat as a single sample )
    # for i in range(2):
    #     print(dgen.__next__()[1])



    resnet = build_resnet()

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    # adam = optimizers.Adam()

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer='adam',  # switch to adam later
        metrics=['accuracy']
    )
    resnet.summary()

    for e in range(maxepoches):
        print("Epoch {}".format(e))
        train_generator = data_generator("train", c.tiles_per_dim)
        step = 0
        for X_batch, y_batch in train_generator:
            # print(X_batch.shape)
            # print(y_batch.shape)
            hist = resnet.train_on_batch(X_batch, y_batch)#, batch_size, epochs=maxepoches)
            preds = resnet.predict_on_batch(X_batch)

            if step % 5 == 0:
                print(hist)
            if step % 10 == 0:
                print(np.argmax(preds, 1)-np.array(y_batch))
            step += 1

        # Validating at end of epoch
        val_steps_max = 0
        prev_total_loss = np.inf
        print("VALIDATING")
        val_generator = data_generator("val", c.tiles_per_dim)
        for X_batch_val, y_batch_val in val_generator:
            hist_val = resnet.test_on_batch(X_batch_val, y_batch_val)
            current_total_loss = hist_val[0]
            if current_total_loss < prev_total_loss:
                resnet.save_weights('resnet_weights_maxSize_{}_tilesPerDim_{}_nTilesPerSample_{}_onlyImg_{}.h5'.format(c.max_size, c.tiles_per_dim, c.n_tiles_per_sample, c.only_images))
            prev_total_loss = hist_val[0]
            print(hist_val)
            val_steps_max += 1
            # if val_steps_max == 5:
            #     print("Finished validating on {} batches".format(val_steps_max))
            #     break
            # callbacks=[reduce_lr])

    # resnet_cifar_10_history = resnet.fit_generator(train_generator,
    #                                                steps_per_epoch=n_samples_train // batch_size,
    #                                                epochs=maxepoches,
    #                                                validation_data=val_generator, validation_steps=1)#,
    #                                                # callbacks=[reduce_lr])




