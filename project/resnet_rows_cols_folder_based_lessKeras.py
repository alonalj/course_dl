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
from preprocessor import *
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
import os
from conf import Conf


def get_steps(c, batch_size, data_type):
    relevant_files = get_relevant_files(data_type,c)
    return len(relevant_files) // batch_size


def data_generator(data_type, tiles_per_dim, data_split_dict, batch_size, c, rows_or_cols):
    import random
    from keras.utils import to_categorical

    relevant_files = get_relevant_files(data_type, c)
    while True:
        random.shuffle(relevant_files)
        for i in range(get_steps(c, batch_size, data_type)):
            X_batch = []
            y_batch = []
            for f in relevant_files[batch_size*i:batch_size*(i+1)]:
                if 'DS' in f:
                    continue
                # print(f)
                label = get_row_col_label(f,c,rows_or_cols) #TODO verify for imgs
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


def build_resnet_rows_col(TILES_PER_DIM, SHAPE, weight_decay=1e-3):
    x_in = Input(shape=(SHAPE, SHAPE, 1))
    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1),
               kernel_regularizer=regularizers.l2(weight_decay))(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_tower_2_layer_img_vs_doc(x, 64, 2, False, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 128, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 256, 2, True, weight_decay=weight_decay)
    x = res_tower_2_layer_img_vs_doc(x, 512, 2, True, weight_decay=weight_decay)

    x = GlobalAveragePooling2D()(x)
    x = Dense(TILES_PER_DIM, activation='softmax')(x)
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



def run(c, rows_or_cols):

    rows_or_cols = rows_or_cols
    print("STARTING {}".format(rows_or_cols))
    tiles_per_dim = c.tiles_per_dim
    is_image = c.is_images

    # resnet_rows_cols = build_resnet_rows_col(tiles_per_dim, c.max_size)

    batch_size = 110
    # TODO: check withoutval in row below

    steps_per_epoch = get_steps(c, batch_size, "train")#len(os.listdir("{}_{}_isImg_{}/0/".format(rows_or_cols, tiles_per_dim, c.is_images)))*tiles_per_dim // batch_size
    maxepoches = 500
    learning_rate = 0.0001
    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=0.00001,verbose=1)

    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    datagen_img_vs_doc_train = data_generator('train', c.tiles_per_dim, '', batch_size, c, rows_or_cols)#ImageDataGenerator()#preprocessing_function=to_grayscale)

    datagen_img_vs_doc_val = data_generator('val', c.tiles_per_dim, '', batch_size, c, rows_or_cols)#ImageDataGenerator()#preprocessing_function=to_grayscale)

    from keras.applications.resnet50 import ResNet50
    resnet_rows_cols = ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(c.max_size,c.max_size,1),
                                       pooling=None, classes=c.tiles_per_dim)
    # Add final layers
    x = resnet_rows_cols.output
    x = Flatten()(x)
    predictions = Dense(c.tiles_per_dim, activation='softmax', name='fc1000')(x)

    # This is the model we will train
    model = Model(inputs=resnet_rows_cols.input, outputs=predictions)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # model.summary()

    # resnet_rows_cols.load_weights('model_weights_{}_{}_isImg_{}.h5'.format(rows_or_cols, tiles_per_dim, is_image))

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
    # model_net_name = 'w.h5'.format(rows_or_cols, tiles_per_dim, is_image)
    # resnet_rows_cols.save(model_net_name)

    # ckpt = keras.callbacks.ModelCheckpoint('model_weights_{}_{}_isImg_{}.h5'.format(rows_or_cols, tiles_per_dim, is_image), monitor='val_acc',
    #                                 verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    # early_stop = keras.callbacks.EarlyStopping('val_acc',min_delta=0.001,patience=120)
    weights_name_format = 'all_weights_train_L{}_A{}_val_L{}_A{}'

    load_weights = False
    if load_weights:
        d = load_obj('all_weights_train_L0.0_A0.0_val_L1.13_A0.82')
        for l_ix in range(len(model.layers)):
            model.layers[l_ix].set_weights(d[l_ix])
        print("loaded")

    baseline_loss = np.inf
    baseline_acc = -np.inf
    count_plateau = 0
    tolerance_plateau = 30
    for e in range(maxepoches):
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

        print("Validating")
        val_steps = 3
        avg_loss_val, avg_acc_val = 0, 0
        for X_batch, y_batch in datagen_img_vs_doc_val:
            batch_loss, batch_acc = model.evaluate(X_batch, y_batch)
            avg_loss_val += batch_loss / float(val_steps)
            avg_acc_val += batch_acc / float(val_steps)
            val_steps_count += 1
            if val_steps_count == val_steps:
                break
        print("avg val loss, acc:", round(avg_loss_val,2), round(avg_acc_val,2))
        if avg_loss_val < baseline_loss and avg_acc_val > baseline_acc:
            print("Saving model, loss change: {} --> {}, acc change: {} --> {}"
                  .format(round(baseline_loss,2), round(avg_loss_val,2), round(baseline_acc,2), round(avg_acc_val,2)))
            all_weights = []
            for l in model.layers:
                all_weights.append(l.get_weights())
            save_obj(all_weights, weights_name_format.format(round(avg_loss,2), round(avg_acc,2),round(avg_loss_val,2), round(avg_acc_val,2)))
            baseline_loss, baseline_acc = avg_loss_val, avg_acc_val
        else:
            count_plateau += 1
        if count_plateau == tolerance_plateau:
            print("No improvement for {} epochs. Moving on.".format(count_plateau))
            return



    # resnet_rows_cols_hist = model.fit_generator(datagen_img_vs_doc_train, validation_data=datagen_img_vs_doc_val,
    #                                                        steps_per_epoch=steps_per_epoch, validation_steps=3, epochs=maxepoches)




    # # resnet_rows_cols.save_weights('model_weights_{}_{}_isImg_{}.h5'.format(rows_or_cols, tiles_per_dim, is_image))
    # # np.save('w.pkl', model.get_weights())
    # # model.set_weights(np.load('w.pkl'))
    # files = os.listdir('example_docs/')
    # files.sort()
    # print(files)  # TODO: remove
    # images = []
    # for f in files:
    #     if 'DS' in f:
    #         continue
    #     im = cv2.imread('example_docs/' + f)
    #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #     im = preprocess_image(im, c)
    #     images.append(im)
    # res = model.predict_on_batch(np.array(images))#, steps=10)
    # print(np.argmax(res,1))
    #
    #
    #
    #
