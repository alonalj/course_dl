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
            for f in relevant_files[batch_size*i:batch_size*(i+1)]:
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


def build_model(c, weights=False):
    from keras.applications.resnet50 import ResNet50
    resnet_rows_cols = ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(c.max_size, c.max_size, 1),
        pooling=None, classes=c.tiles_per_dim)
    # Add final layers
    x = resnet_rows_cols.output
    x = Flatten()(x)
    predictions = Dense(c.tiles_per_dim, activation='softmax', name='fc1000',kernel_regularizer=keras.regularizers.l2(0.0001))(x)

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

    batch_size = 128
    steps_per_epoch = get_steps(c, batch_size, "train")
    max_epochs = 900

    datagen_img_vs_doc_train = data_generator('train', batch_size, c, rows_or_cols)
    datagen_img_vs_doc_val = data_generator('val', batch_size, c, rows_or_cols)

    # model = build_model(c, weights='weights_img_True_t_4_rows_L0.21_A0.94_val_L0.27_A0.91')
    model = build_model(c)
    weights_name_format = 'weights_img_{}_t_{}_{}'.format(c.is_images, c.tiles_per_dim, rows_or_cols)
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
                for l in model.layers:
                    all_weights.append(l.get_weights())
                save_obj(all_weights, weights_name_format+'_L{}_A{}_val_L{}_A{}'
                         .format(round(avg_loss,2), round(avg_acc,2),round(avg_loss_val,2), round(avg_acc_val,2)))
                baseline_loss, baseline_acc = avg_loss_val, avg_acc_val
                count_plateau = 0
                if avg_acc_val > 0.9:
                    print("*** VAL BETTER THAN 0.9 :) MOVING ON... ***")
                    return
            else:
                count_plateau += 1
                print("No val improvement since loss, acc:", baseline_loss, baseline_acc)
            if count_plateau == tolerance_plateau and avg_loss < 0.01:
                print("No improvement for {} epochs. Moving on.".format(count_plateau))
                return

    else:
        _evaluate('example/')


