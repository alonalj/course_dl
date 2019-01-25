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
    split_dict = load_obj(data_split_dict)
    folders = split_dict[data_type]

    if 'doc' in data_split_dict:
        dataset_folder = "dataset_{}_isImg_False".format(tiles_per_dim)
    else:
        dataset_folder = "dataset_{}_isImg_True".format(tiles_per_dim)
    # while True:
    np.random.shuffle(folders)  # shuffle folders between epochs
    X_batch = []
    y_batch = []
    noise = False
    if data_type == 'train':
        if 'doc' not in data_split_dict:
            crop_start_w = range(0, 46, 15)
            crop_start_h = range(0, 46, 15)
        else:
            crop_start_w = range(0, 91, 30)
            crop_start_h = range(0, 91, 30)
        c_w = random.choice(crop_start_w)
        c_h = random.choice(crop_start_h)
    else:
        crop_start_w = [0]
        crop_start_h = [0]
        c_w = 0
        c_h = 0
    folders = [f for f in folders if '_crw_'+str(c_w)+'_crh_'+str(c_h) in f]
    for folder in folders:
        original_images = []
        processed_images = []
        skip_folder = False
        flip_img = False
        if random.random() > 0.5 and data_type == 'train':
            flip_img = True
        if random.random() > 0.6 and data_type == 'train':
            noise = True
        if c.is_images and len(folder) < 7:
            # print(len(folder))
            continue
        folder_path = dataset_folder + '/' + folder
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
                img_shape = im.shape[0] + im.shape[1]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im_resized = resize_image(im, max_size=c.max_size, simple_reshape=True)
            except:
                print("failed on {}".format(folder_path + '/' + f))  # TODO: remove
                skip_folder = True
                continue

            if noise:
                gauss = get_gauss_noise(im_resized)
                im_resized = noisy("gauss",im_resized, gauss)

            if im_resized.shape != (c.max_size, c.max_size):
                print("Bad shape for folder {}, file {}".format(folder, f))

            if flip_img:
                im_resized = cv2.flip(im_resized, 1)
            images_in_folder.append(im_resized)
            original_images.append(im)

        if np.array(images_in_folder).shape != (c.n_tiles_per_sample, c.max_size, c.max_size):
            continue
        # print(np.array(images_in_folder).shape)
        images_in_folder = add_similarity_channel(images_in_folder, original_images, c)

        X_batch.append(np.array(images_in_folder))  # a folder is one single sample
        # print(labels_in_folder)
        folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
        y_batch.append(folder_labels)
        if len(y_batch) == batch_size:
            # print(np.array(X_batch).ndim)
            # print(np.array(X_batch))
            if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size, 2):
                print(folder)
                print(np.array(X_batch).shape)
            yield list(np.array(X_batch).reshape(c.n_tiles_per_sample, batch_size, c.max_size, c.max_size, 2)), \
                  list(np.array(y_batch).reshape(c.n_tiles_per_sample, batch_size, c.n_classes))
            X_batch = []
            y_batch = []
    # handle last batch in case n_folders not fully divisible by batch_size (has a remainder)
    if len(y_batch) != batch_size:  # if equal, already yielded above
        # print(np.array(X_batch).shape)
        # print(np.array(X_batch))
        if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size, 2):
            print(folder)
            print(np.array(X_batch).shape)
        yield list(np.array(X_batch).reshape(c.n_tiles_per_sample, -1, c.max_size, c.max_size, 2)), \
              list(np.array(y_batch).reshape(c.n_tiles_per_sample, -1, c.n_classes))


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

def run(c):

    batch_size = 128
    # adam = optimizers.Adam()
    if c.n_tiles_per_sample > 6:
        batch_size = 50
    if c.n_tiles_per_sample > 20:
        batch_size = 35
        # adam = optimizers.Adam(0.0001)

    maxepoches = 250
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
    labels = []

    for e in range(maxepoches):
        print("Epoch {}".format(e))
        train_generator = data_generator("train", c.tiles_per_dim, c.data_split_dict, batch_size, c)
        step = 0
        for X_batch, y_batch in train_generator:
            # print(X_batch.shape)
            # print(y_batch.shape)
            hist = resnet.train_on_batch(X_batch, y_batch)  # , batch_size, epochs=maxepoches)
            preds = resnet.predict_on_batch(X_batch)

            if step % 5 == 0:
                print(hist)
            if step % 10 == 0:
                preds = np.array(preds)
                y = np.array(y_batch)
                # print(preds - y)
                assert preds.shape == y.shape
            step += 1

        # Validating at end of epoch
        print("VALIDATING")
        val_generator = data_generator("val", c.tiles_per_dim, c.data_split_dict, batch_size, c)
        current_acc = []
        for X_batch_val, y_batch_val in val_generator:
            hist_val = resnet.test_on_batch(X_batch_val, y_batch_val)
            current_acc.append(np.mean(hist_val[-c.n_tiles_per_sample:]))
        print("val labels best so far", labels)
        current_avg_acc = np.mean(current_acc)
        if current_avg_acc > best_avg_acc_val:
            labels = []
            resnet.save_weights(
                'resnet_maxSize_{}_tilesPerDim_{}_nTilesPerSample_{}_isImg_{}_mID_{}_L_{}.h5'.format(c.max_size,
                                                                                                     c.tiles_per_dim,
                                                                                                     c.n_tiles_per_sample,
                                                                                                     c.is_images,
                                                                                                     c.mID,
                                                                                                     str(
                                                                                                         current_avg_acc)))

            print(hist_val)
            best_avg_acc_val = current_avg_acc
            print("best avg acc val: {}".format(best_avg_acc_val))
            no_improvement_counter = 0  # reset
            logits = resnet.predict_on_batch(X_batch)
            for l in logits:
                idx_max = l.argmax(axis=1)
                idx_max = int(idx_max)
                if idx_max == c.n_classes - 1:
                    # OoD
                    idx_max = -1
                labels.append(idx_max)
            print("val labels best", labels)
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

        print(current_avg_acc)
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
    run(c)
