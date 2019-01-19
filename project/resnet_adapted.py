import keras
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, Input
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

def resnet_weights_shared_over_tiles(c, x_in=None, weight_decay=0.0001):
    # x_in = keras.layers.ZeroPadding2D()(x_in)
    x_in = Input(shape=(c.max_size, c.max_size, 1))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    # x = Dense(c.n_classes, activation='softmax')(x)
    x = Dense(c.n_classes)(x)
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


def build_resnet(c, weight_decay=0.0001):
    # TODO: ? another input is the shape of the image - could immediately tell if it's OoD
    n_tiles_per_sample = c.n_tiles_per_sample  # original tiles + OoD
    # passing all tiles from each batch into the conv (a batch contains multiple folders, from each folder we want
    # the evaluation over all tiles to happen in the same pass)
    inputs_from_sample = []
    outputs_from_sample = []
    shared_net = resnet_weights_shared_over_tiles(c)
    for i in range(n_tiles_per_sample):
        x_in = keras.layers.Input(shape=(c.max_size, c.max_size), name="in_{}".format(i))
        inputs_from_sample.append(x_in)
        x_in = keras.layers.Reshape(target_shape=(x_in.shape[1], x_in.shape[2], 1),name="in_reshape_{}".format(i))(x_in)
    # for i in range(n_tiles_per_sample):
        x_out = shared_net(x_in)
        outputs_from_sample.append(x_out)
    # layer adds all output vectors

    # sum = Input(tensor=keras.backend.constant([0]))
    # Making sure each position of the t^2 options is associated with exactly one tile
    # (creating a penalty by summing over all outputs for sample (all tiles' outputs)) over the relevant t^2 classes
    for tile_idx in range(len(outputs_from_sample)):
        outputs_from_sample_original = keras.layers.Lambda(lambda x: x[:, :c.n_original_tiles])(outputs_from_sample[tile_idx])
        outputs_from_sample_OoD = keras.layers.Lambda(lambda x: x[:, c.n_original_tiles:])(
            outputs_from_sample[tile_idx])
        if tile_idx == 0:
            sum_original_tile_preds = outputs_from_sample_original
            sum_OoD_tile_preds = outputs_from_sample_OoD
        else:
            sum_original_tile_preds = keras.layers.Add()([outputs_from_sample_original, sum_original_tile_preds])   # shape(batch_size, n_original_tiles) (we summed over tiles, i.e. n_classes)
            sum_OoD_tile_preds = keras.layers.Add()([outputs_from_sample_OoD,
                                                     sum_OoD_tile_preds])
            # if indeed we have exactly one prediction per position over all tiles then we should get sum = [1,1,..,1,1], e.g. for t=2 we get [1,1,1,1] (Note - there is no limit on OoD)


    # This is basically doing sum(1 - sum_original_tile_preds) - ideally would be sum(1-1)=0 if each position were
    # associated with exactly one tile. A bad case would be all original tiles' predictions are the same:
    # -> for t=2 we would get sum = 1-4 = -3 (instead of 1-1=0)
    # simple example to understad computation here: if we replace (1-x) below with (5-x*0) we get a tensor of shape (batch_size, n_original_tiles) with all values = 25. Then we sum accross n_original_tiles and obtain ±100 in the case of t=2 (4 original tiles)
    diff_original = keras.layers.Lambda(lambda x: (1-x)**2)(sum_original_tile_preds)  # shape= (batch_size, n_original_tiles).  Subtracting total sum of prds over tiles from 1, so if many tiles got 1 in the same idx, we have 1-(large_number)
    for original_tile_class_idx in range(c.n_original_tiles):
        original_tile_diff = keras.layers.Lambda(lambda x: x[:, original_tile_class_idx])(diff_original)
        if original_tile_class_idx == 0:
            sum_diff_all_tiles_in_sample = original_tile_diff
        else:
            sum_diff_all_tiles_in_sample = keras.layers.Add()([original_tile_diff, sum_diff_all_tiles_in_sample])  # shape=(batch_size, 1)
    # t=2, if all get same pred: (1-x)**2 = [1,1,1-4,1]**2 = 1+1+9+1 = 12 --> sum over all original tiles = 12*4 = 28 -->
    # multiply by coefficient, say 0.01 (below) we get 0.28. --> Subtract this penalty from all predictions (resnet out):
    # we get [-0.28, -0.28, 1-0.28, -0.28, -0.28] (all got prediction of 3rd place). This goes to cross-entropy.

    diff_OoD = keras.layers.Lambda(lambda x: (c.tiles_per_dim - x) ** 2)(sum_OoD_tile_preds) # e.g. fot t=2 should be 0 if only two tiles have this prediction
    for OoD_tile_class_idx in range(1):  # always one class for OoD   #range(c.n_classes-c.n_original_tiles):
        OoD_tile_diff = keras.layers.Lambda(lambda x: x[:, OoD_tile_class_idx])(diff_OoD)
        sum_diff_all_tiles_in_sample = keras.layers.Add()([OoD_tile_diff, sum_diff_all_tiles_in_sample])  # shape=(batch_size, 1)

    # adding the sum_diff_all_tiles_in_sample penalty with small weight to actual outputs so cross entropy will be harmed if not all diff_original
    penalized_outputs_from_sample = []
    penalty = keras.layers.Lambda(lambda x: 0.05 * x)(sum_diff_all_tiles_in_sample)
    # penalty = sum_diff_all_tiles_in_sample
    for o in outputs_from_sample:
        o = keras.layers.Subtract()([o, penalty])
        o = Dense(c.n_classes, activation='softmax')(o)
        penalized_outputs_from_sample.append(o)

    print(inputs_from_sample)
    print(penalized_outputs_from_sample)

    return Model(inputs=inputs_from_sample, outputs=penalized_outputs_from_sample)


