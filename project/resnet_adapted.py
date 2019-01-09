import keras
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, Input
from keras.layers import Flatten, InputLayer, BatchNormalization, Activation
from keras.models import Model
from keras import regularizers
from conf import Conf


c = Conf()

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

def pass_single_tile_per_sample(x_in, weight_decay=0.0001):
    x_in = keras.layers.ZeroPadding2D()(x_in)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    x = Dense(c.n_classes, activation='softmax')(x)
    return x


def build_resnet(weight_decay=0.0001):
    n_tiles_per_sample = c.n_tiles_per_sample  # original tiles + OoD
    x_in_all_tiles = Input(shape=(n_tiles_per_sample, c.max_size, c.max_size))
    # passing all tiles from each batch into the conv (a batch contains multiple folders, from each folder we want
    # the evaluation over all tiles to happen in the same pass)
    for i in range(n_tiles_per_sample):
        x_in = keras.layers.Lambda(lambda x: x[:,i, :, :])(x_in_all_tiles)
        x_in = keras.layers.Reshape(target_shape=(x_in.shape[1], x_in.shape[2], 1))(x_in)
        x_out_all_tiles = pass_single_tile_per_sample(x_in)
        if i != 0:
            x_out_all_tiles = keras.layers.merge.concatenate([x_out_prev, x_out_all_tiles])
        x_out_prev = x_out_all_tiles
        # TODO: add regularization so that x_out_prev vectors concatenated are orthogonal except for -1s (might need two additional labels to represent -1 instead of just one)
    return Model(inputs=x_in_all_tiles, outputs=x_out_all_tiles)


