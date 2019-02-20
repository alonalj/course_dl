from keras import optimizers

from preprocessor import *
import cv2
import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
import numpy as np
import os
from conf import Conf
import cv2
import keras
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, Input
from keras.layers import Flatten, InputLayer, BatchNormalization, Activation
from keras.models import Model
from keras import regularizers, optimizers
from keras.regularizers import Regularizer
from keras import backend as K
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from conf import Conf

def get_steps(c, batch_size, data_type):
    relevant_files = get_relevant_files(data_type,c)
    return max(len(relevant_files) // batch_size, 1)


def data_generator(data_type, batch_size, c, rows_or_cols):
    import random
    from keras.utils import to_categorical

    relevant_files = get_relevant_files(data_type, c)
    while True:
        random.shuffle(relevant_files)
        for i in range(get_steps(c, batch_size, data_type)):
            X_batch = []
            y_batch = []
            for f in relevant_files[batch_size*i:batch_size*(i+1)]:  # TODO: make balanced precisely?
                if 'DS' in f:
                    continue
                # print(f)
                label = get_row_col_label(f,c,rows_or_cols=="rows") #TODO verify for imgs
                im = cv2.imread(c.output_dir+f)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im = preprocess_image(im, c)
                if im.shape != (c.max_size, c.max_size, 1):
                    continue
                X_batch.append(im)
                y_batch.append(to_categorical(label, num_classes=c.tiles_per_dim))

            combined_images, labels = np.array(X_batch), np.array(y_batch)
            # print(labels)
            if len(combined_images) > 0:
                yield combined_images, labels


def _evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    print(files)  #TODO: remove
    images = []
    for f in files:
        if 'DS' in f:
            continue
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = _predict(images)
    print(Y)  # TODO - remove!
    return Y


def res_2_layer_block_img_vs_doc(x_in, dim, downsample=False, weight_decay=0.0001):
    x = Conv2D(dim, kernel_size=(3, 3), padding='same', strides=(2, 2) if downsample else (1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(dim, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    if downsample:
        x_in = Conv2D(dim, kernel_size=(1, 1), padding='same', strides=(2, 2),
                      kernel_regularizer=regularizers.l2(weight_decay))(x_in)

    x = keras.layers.add([x, x_in])
    x = Activation('relu')(x)

    return x


def res_tower_2_layer_img_vs_doc(x, dim, num_layers, downsample_first=True, weight_decay=0.0001):
    for i in range(num_layers):
        x = res_2_layer_block_img_vs_doc(x, dim, downsample=(i == 0 and downsample_first), weight_decay=weight_decay)
    return x


def res_3_layer_block_img_vs_doc(x_in, dim_reduce, dim_out, downsample=False, adjust_skip_dim=False, weight_decay=0.0001):
    x = Conv2D(dim_reduce, kernel_size=(1, 1), padding='same', strides=(2, 2) if downsample else (1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(dim_reduce, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(dim_out, kernel_size=(1, 1), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    if downsample or adjust_skip_dim:
        x_in = Conv2D(dim_out, kernel_size=(1, 1), padding='same', strides=(2, 2) if downsample else (1, 1),
                      kernel_regularizer=regularizers.l2(weight_decay))(x_in)

    x = keras.layers.add([x, x_in])
    x = Activation('relu')(x)

    return x


def res_tower_img_vs_doc(x, dim, num_layers, downsample_first=True, adjust_first=False, weight_decay=0.0001):
    for i in range(num_layers):
        x = res_3_layer_block_img_vs_doc(x, int(dim / 4), dim, downsample=(i == 0 and downsample_first),
                                         adjust_skip_dim=(i == 0 and adjust_first), weight_decay=weight_decay)
    return x

def build_model(c, weights=False, weight_decay=1e-3):
    x_in = Input(shape=(c.max_size, c.max_size, 1))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer_img_vs_doc(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    x = Dense(c.tiles_per_dim, activation='softmax')(x)
    model = Model(inputs=x_in, outputs=x)
    if weights:
        model.load_weights(weights)
        print("loaded weights {}".format(weights))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


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

# def to_grayscale(im):
#     return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)



def _predict(images):
    c = Conf()
    c.tiles_per_dim = 4
    c.is_images = True
    c.max_size = 112
    images = add_similarity_channel(images,images,c,sim_on_side=True)
    processed_images = []
    for im in images:
        im = preprocess_image(im, c)
        processed_images.append(im)
    model = build_model(c,'weights_img_True_t_4_rows_L0.21_A0.94_val_L0.27_A0.91')
    res = model.predict_on_batch(np.array(processed_images))  # , steps=10)
    print(np.argmax(res, 1))


def run(c, rows_or_cols):

    batch_size = 110
    steps_per_epoch = get_steps(c, batch_size, "train")
    max_epochs = 1000

    datagen_img_vs_doc_train = data_generator('train', batch_size, c, rows_or_cols)
    datagen_img_vs_doc_val = data_generator('val', batch_size, c, rows_or_cols)

    # model = build_model(c, weights='weights_no_sim_img_True_t_4_cols_L7.45_A0.25_val_L4.19_A1.0')
    model = build_model(c)
    weights_name_format = 'weights_no_sim_img_{}_t_{}_{}'.format(c.is_images, c.tiles_per_dim, rows_or_cols)
    train = True

    if train:
        print("STARTING")
        baseline_loss = np.inf
        baseline_acc = -np.inf
        count_plateau = 0
        tolerance_plateau = 40
        for e in range(max_epochs):
            train_steps_count, val_steps_count = 0, 0
            avg_loss, avg_acc = 0, 0
            print("Epoch {}".format(e))
            for X_batch, y_batch in datagen_img_vs_doc_train:
                loss, acc = model.train_on_batch(X_batch, y_batch)
                avg_loss += loss / float(steps_per_epoch)
                avg_acc += acc / float(steps_per_epoch)
                train_steps_count += 1
                if train_steps_count == steps_per_epoch:
                    break
            print("Train loss, acc:", round(avg_loss,2), round(avg_acc,2))
            print("Validating")
            val_steps = get_steps(c, batch_size, "val")
            avg_loss_val, avg_acc_val = 0, 0
            for X_batch, y_batch in datagen_img_vs_doc_val:
                batch_loss, batch_acc = model.evaluate(X_batch, y_batch)
                avg_loss_val += batch_loss / float(val_steps)
                avg_acc_val += batch_acc / float(val_steps)
                val_steps_count += 1
                if val_steps_count == val_steps:
                    break
            print("Val loss, acc:", round(avg_loss_val,2), round(avg_acc_val,2))
            # if avg_loss_val < baseline_loss and avg_acc_val > baseline_acc:
            if avg_acc_val > baseline_acc:
                print("Saving model, loss change: {} --> {}, acc change: {} --> {}"
                      .format(round(baseline_loss,2), round(avg_loss_val,2), round(baseline_acc,2), round(avg_acc_val,2)))
                all_weights = []
                model.save_weights(weights_name_format+'_L{}_A{}_val_L{}_A{}'
                         .format(round(avg_loss,2), round(avg_acc,2),round(avg_loss_val,2), round(avg_acc_val,2)))
                # for l in model.layers:
                #     all_weights.append(l.get_weights())
                # save_obj(all_weights, weights_name_format+'_L{}_A{}_val_L{}_A{}'
                #          .format(round(avg_loss,2), round(avg_acc,2),round(avg_loss_val,2), round(avg_acc_val,2)))
                baseline_loss, baseline_acc = avg_loss_val, avg_acc_val
                count_plateau = 0
                if avg_acc_val > 0.9:
                    print("*** VAL BETTER THAN 0.9 :) MOVING ON... ***")
                    return
            else:
                count_plateau += 1
                print("No val improvement since loss, acc:", baseline_loss, baseline_acc)
            if count_plateau >= tolerance_plateau and avg_loss <= 0.05:
                print("No improvement for {} epochs. Moving on.".format(count_plateau))
                return

    else:
        _evaluate('example/')


