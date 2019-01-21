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

def res_2_layer_block(x_in, dim, downsample=False, weight_decay=0.0001):
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


def res_tower_2_layer(x, dim, num_layers, downsample_first=True, weight_decay=0.0001):
    for i in range(num_layers):
        x = res_2_layer_block(x, dim, downsample=(i == 0 and downsample_first), weight_decay=weight_decay)
    return x


def res_3_layer_block(x_in, dim_reduce, dim_out, downsample=False, adjust_skip_dim=False, weight_decay=0.0001):
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


def res_tower(x, dim, num_layers, downsample_first=True, adjust_first=False, weight_decay=0.0001):
    for i in range(num_layers):
        x = res_3_layer_block(x, int(dim / 4), dim, downsample=(i == 0 and downsample_first),
                              adjust_skip_dim=(i == 0 and adjust_first), weight_decay=weight_decay)
    return x

def build_resnet_cifar_10(weight_decay):
    x_in = Input(shape=(32, 32, 3))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=x_in, outputs=x)


resnet_img_vs_doc = build_resnet_cifar_10(1e-3)

batch_size = 128
maxepoches = 2
learning_rate = 0.1


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


reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

# sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

resnet_img_vs_doc.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
resnet_img_vs_doc.summary()

datagen = ImageDataGenerator()
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

# keras.callbacks.ModelCheckpoint('is_img_or_doc.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


resnet_cifar_10_history = resnet_img_vs_doc.fit_generator(datagen.flow_from_directory('img_vs_doc', target_size=(32, 32),
                                                                                      batch_size=batch_size),
                                                          steps_per_epoch= (80505+20160) // batch_size,
                                                          epochs=maxepoches,
                                                          validation_steps = 10,
                                                          validation_data=datagen.flow_from_directory('img_vs_doc_val', target_size=(32, 32)))
                                                        # callbacks=[reduce_lr])

resnet_img_vs_doc.save_weights('is_img_or_doc.h5')