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
import os
from conf import Conf


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


def build_resnet_ood(SHAPE, weight_decay=1e-3):
    x_in = Input(shape=(SHAPE, SHAPE*2, 1))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer_img_vs_doc(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=x_in, outputs=x)


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
    if epoch > 5:
        lr = 0.01
    if epoch > 10:
        lr = 0.001
    if epoch > 20:
        lr = 1e-4
    return lr

# def to_grayscale(im):
#     return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def build_model(c, weights=False):
    from keras.applications.resnet50 import ResNet50
    from preprocessor import load_obj
    resnet_rows_cols = ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(c.max_size, c.max_size, 1),
        pooling=None, classes=c.tiles_per_dim)
    # Add final layers
    x = resnet_rows_cols.output
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax', name='fc1000',kernel_regularizer=keras.regularizers.l2(0.0001))(x)

    # This is the model we will train
    model = Model(inputs=resnet_rows_cols.input, outputs=predictions)

    # sgd = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True, decay=0.00000001)
    adam = optimizers.adam(lr=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy']
    )
    if weights:
        d = load_obj(weights)
        for l_ix in range(len(model.layers)):
            model.layers[l_ix].set_weights(d[l_ix])
        print("loaded weights")
    return model


def run(c):

    import glob

    is_image = c.is_images
    SHAPE = c.max_size

    # from resnet_ood_pairs import *
    # resnet_ood = build_resnet(c.max_size, c.tiles_per_dim)
    resnet_ood = build_model(c)#build_resnet_ood(SHAPE)

    batch_size = 128
    path = "ood_isImg_{}".format(c.is_images)
    train_len = len(glob.glob(path+'/0'+'/*')) * 2

    steps_per_epoch = train_len // batch_size

    reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=50, min_lr=0.00001)

    resnet_ood.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    resnet_ood.summary()

    datagen_ood_train = ImageDataGenerator(preprocessing_function=lambda x: x / 255.)#preprocessing_function=to_grayscale)
    datagen_ood_Val = ImageDataGenerator(preprocessing_function=lambda x: x / 255.)#preprocessing_function=to_grayscale)

        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False)  # randomly flip images

    # datagen.fit(X_train)
    # if os.path.exists('model_ood_pairs_isImg_{}.h5'.format(is_image)):
    #     print("found weights, loading and continuing to train.")
    #     resnet_ood.load_weights('model_ood_pairs_isImg_{}.h5'.format(is_image))
    ckpt = keras.callbacks.ModelCheckpoint('model_ood_pairs_isImg_{}.h5'.format(is_image), monitor='val_loss',
                                    verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    early_stop = keras.callbacks.EarlyStopping('val_loss',min_delta=0.2,patience=10)

    resnet_rows_cols_hist = resnet_ood.fit_generator(
        datagen_ood_train.flow_from_directory(path,
                                               target_size=(c.max_size, c.max_size),
                                               color_mode='grayscale',
                                               batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=1000,
        validation_steps=30,
        shuffle=True,
        validation_data=
        datagen_ood_Val.flow_from_directory(path+'_val',
                                               target_size=(c.max_size, c.max_size),
                                               color_mode='grayscale'),
        callbacks=[reduce_lr, ckpt])#, early_stop])

    # resnet_ood.save_weights('model_{}_{}.h5'.format(rows_or_cols, tiles_per_dim))


if __name__ == '__main__':
    c = Conf()
    c.is_images = False
    run(c)