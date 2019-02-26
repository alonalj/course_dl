import keras
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, \
    Input
from keras.layers import Flatten, InputLayer, BatchNormalization, Activation
from keras.models import Model
from keras import regularizers
from keras.regularizers import Regularizer
from keras import backend as K
import numpy as np
from conf import Conf


# c = Conf()

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


# def pass_single_image(x_in, weight_decay=0.0001):
#     x = Conv2D(64, kernel_size=(7, 7), padding='same', strides=(2, 2),
#                kernel_regularizer=regularizers.l2(weight_decay))(x_in)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
#
#     x = res_tower(x, 256, 3, False, True)
#     x = res_tower(x, 512, 8, True)
#     x = res_tower(x, 1024, 36, True)
#     x = res_tower(x, 2048, 3, True)
#
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(c.n_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     return x

def resnet_weights_shared_over_tiles(max_size, n_classes, x_in=None, weight_decay=0.0001):
    # x_in = keras.layers.ZeroPadding2D()(x_in)
    x_in = Input(shape=(max_size, max_size, 1))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)

    return Model(x_in, x, name="Resnet_model")


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = keras.backend.cast_to_floatx(l1)
        self.l2 = keras.backend.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += keras.backend.sum(self.l1 * keras.backend.abs(x))
        if self.l2:
            regularization += keras.backend.sum(self.l2 * keras.backend.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# def build_resnet(max_size, n_tiles_per_sample, n_classes, n_original_tiles, tiles_per_dim, nweight_decay=0.0001):
#     # TODO: ? another input is the shape of the image - could immediately tell if it's OoD
#     n_tiles_per_sample = n_tiles_per_sample  # original tiles + OoD
#     # passing all tiles from each batch into the conv (a batch contains multiple folders, from each folder we want
#     # the evaluation over all tiles to happen in the same pass)
#
#     shared_net = resnet_weights_shared_over_tiles(max_size, n_classes)
#
#     x_in_1 = keras.layers.Input(shape=(max_size, max_size, 1), name="in_1")
#     # x_in_1 = keras.layers.Reshape(target_shape=(x_in_1.shape[1], x_in_1.shape[2], 1), name="in_reshape_1")(x_in_1)
#     # for i in range(n_tiles_per_sample):
#     x_out_1 = shared_net(x_in_1)
#
#     x_in_2 = keras.layers.Input(shape=(max_size, max_size, 1), name="in_2")
#     # x_in_2 = keras.layers.Reshape(target_shape=(x_in_2.shape[1], x_in_2.shape[2], 1), name="in_reshape_2")(x_in_2)
#     # for i in range(n_tiles_per_sample):
#     x_out_2 = shared_net(x_in_2)
#
#     concat = keras.layers.concatenate([x_out_1, x_out_2])
#     x = Dense(200, activation='relu')(concat)
#     x = Dense(100, activation='relu')(x)
#     x = Dense(10, activation='relu')(x)
#     x = Dense(2, activation='softmax')(x)
#
#     return Model(inputs=[x_in_1, x_in_2], outputs=x)


def build_resnet(max_size, n_tiles_per_sample, n_classes):
    # TODO: ? another input is the shape of the image - could immediately tell if it's OoD
    n_tiles_per_sample = n_tiles_per_sample  # original tiles + OoD
    # passing all tiles from each batch into the conv (a batch contains multiple folders, from each folder we want
    # the evaluation over all tiles to happen in the same pass)

    shared_net = resnet_weights_shared_over_tiles(max_size, n_classes)

    x_in_1 = keras.layers.Input(shape=(max_size, max_size, 1), name="in_1")
    # x_in_1 = keras.layers.Reshape(target_shape=(x_in_1.shape[1], x_in_1.shape[2], 1), name="in_reshape_1")(x_in_1)
    # for i in range(n_tiles_per_sample):
    x_out_1 = shared_net(x_in_1)

    x_in_2 = keras.layers.Input(shape=(max_size, max_size, 1), name="in_2")
    # # x_in_2 = keras.layers.Reshape(target_shape=(x_in_2.shape[1], x_in_2.shape[2], 1), name="in_reshape_2")(x_in_2)
    # # for i in range(n_tiles_per_sample):
    x_out_2 = shared_net(x_in_2)

    concat = keras.layers.concatenate([x_out_1, x_out_2])
    x = Dense(200, activation='relu')(concat)
    x = Dense(100, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    return Model(inputs=[x_in_1, x_in_2], outputs=[x])


def single_resnet(c):
    from keras.applications.resnet50 import ResNet50

    resnet_rows_cols = ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(c.max_size, c.max_size, 1),
        pooling=None, classes=c.tiles_per_dim)
    # Add final layers
    x = resnet_rows_cols.output
    x = Flatten()(x)

    return Model(inputs=resnet_rows_cols.input, outputs=x)

def build_resnet(c, weights=False):

    # predictions = Dense(c.tiles_per_dim, activation='softmax', name='fc1000',kernel_regularizer=keras.regularizers.l2(0.0001))(x)

    # This is the model we will train
    # model = Model(inputs=resnet_rows_cols.input, outputs=x)
    # model = single_resnet(c)
    x_in_1 = keras.layers.Input(shape=(c.max_size, c.max_size, 1), name="in_1")
    x_in_2 = keras.layers.Input(shape=(c.max_size, c.max_size, 1), name="in_2")

    r = single_resnet(c)

    x_out_1 = r(x_in_1)
    x_out_2 = r(x_in_2)

    concat = keras.layers.concatenate([x_out_1, x_out_2])
    x = Dense(200, activation='relu')(concat)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[x_in_1, x_in_2], outputs=[x])

    from keras import optimizers

    sgd = optimizers.SGD(lr=0.0000001, momentum=0.9, nesterov=True, decay=0.0001)
    # adam = optimizers.adam(lr=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    # if weights:
    #     d = load_obj(weights)
    #     for l_ix in range(len(model.layers)):
    #         model.layers[l_ix].set_weights(d[l_ix])
    #     print("loaded weights")
    return model
