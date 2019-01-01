from keras.layers import Dense, Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras import Model
from keras import backend as K
import tensorflow as tf
import os
import warnings

from cifar100vgg import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def download_resulting_weights_for_transfer_learning():
    try:
        import urllib.request
    except:
        warnings.warn("****** urllib package not installed - canott fetch the solutions (i.e. the resulting weights post transfer-learning).******")
    try:
        # Resulting weights for section "Fine tunning"
        urllib.request.urlretrieve("https://drive.google.com/open?id=16xSArtqC9m7MrvFoxYKBcmCW1vyUEsqi",
                                   'cifar10vgg_nSamples_100.h5')
        urllib.request.urlretrieve("https://drive.google.com/open?id=1bvQ5dzg7BelIrN4uuAFYUPq3oYTkNUNq",
                                   'cifar10vgg_nSamples_1000.h5')
        urllib.request.urlretrieve("https://drive.google.com/open?id=1koxj7XFAozl6cjLArCG7h-iZ58PQfsdk",
                                   'cifar10vgg_nSamples_10000.h5')
        # Resulting weights for section "Your own solution"
        urllib.request.urlretrieve("https://drive.google.com/open?id=1iHyCkEza0z4DIV6h587C0JjAk-F1RPZb",
                                   'cifar10vgg_reg_0_nSamples_10000.h5')
        urllib.request.urlretrieve("https://drive.google.com/open?id=17Y4L8_3T-Y339DCr8c7gohlqu35PXcxO",
                                   'cifar10vgg_reg_0.5_nSamples_10000.h5')
        urllib.request.urlretrieve("https://drive.google.com/open?id=1zCWsaoneJztcS_aspUc8hO2Ci_QYO0jo",
                                   'cifar10vgg_reg_5_nSamples_10000.h5')
    except:
        warnings.warn("****** Cannot auto-download the solution weights (post trasnfer learning weights). If you want to load our final results, please use links in code to download manually. ******")


def try_loading_saved_weights(model, weights_file_name):
    import os
    if os.path.exists(weights_file_name):
        model.load_weights(weights_file_name)
        print("Successfully loaded weights from: {}".format(weights_file_name))
    else:
        try:
            download_resulting_weights_for_transfer_learning()
            model.load_weights(weights_file_name)
            print("Successfully loaded weights from: {}".format(weights_file_name))
        except:
            print("ERROR")
            pass
    return model


def plot_figures(dict_x_y, title, metrics=['loss', 'acc'], iterations=None, x_axis_name='Epochs'):
    """
    the results should be depicted in two figures (one for loss
    and one for accuracy), where the X-axis is the number of iterations (number of
    backward iterations during training) and the Y -axis will be the loss or accuracy.
    In each figure show both the training and validation curves.
    :param dict_x_y - a dictionary of the form {'x_axis_name': [], 'y_axis_name': []} corresponding to
            the x and y values to plot. E.g.: {'Steps': [1,2,..5], 'Accuracy': [89, 95,..., 100]}
    :param title - plot title
    """
    import pandas as pd

    def plot_df(df, title, color):
        # plot using a data frame
        x_name = df.columns[0]
        y_name = df.columns[1]

        plt.plot(x_name, y_name, data=df, marker='', color=color, linewidth=2)
        plt.legend()
        plt.xlabel(x_name)
        plt.ylabel(y_name.split(' ')[1].title())
        plt.title(title)

    for metric in metrics:
        for data_type in ['', 'val_']:
            data_type_name = 'val' if 'val' in data_type else 'train'
            color = 'c' if 'val' in data_type else 'darkslategray'
            # gather metric results from all epochs
            metric_values = dict_x_y[data_type + metric]
            if iterations is None:
                iterations = range(len(dict_x_y[data_type + metric]))  # num epochs or whatever dict len represents

            # plot metric results
            if metric_values[0] is not None:
                df = pd.DataFrame.from_dict({x_axis_name: iterations, data_type_name + ' ' + metric: metric_values})
                plot_df(df, title, color)
        # display and close so next metric type is on new plot
        plt.savefig('{}.png'.format(metric + ' for ' + title))
        # plt.show()
        plt.close()


class cifar10vgg(cifar100vgg):
    '''
    Creates a new DNN according to guidelines in 3.1 by inheriting from cifar100vgg model supplied in the github link.
    Original guidelines: Replace the last fully-connected layer with a new
    initialized layer, where the output size is the number of classes in CIFAR-10 (i.e.,
    10 classes). We will then freeze all other layers and train the last layer only with
    the small (sample) available dataset for CIFAR-10. Repeat this procedure for
    datasets sampled from CIFAR-10 of sizes 100, 1000, 10000. Make sure you use
    the random seed 42 so we can reproduce your results.
    '''

    def __init__(self, train=False, first_trainable_layer=-2, add_regularizer_for_cifar100_weights=False):
        super().__init__(train)
        self.num_classes = 10
        self.add_regularizer_for_cifar100_weights = add_regularizer_for_cifar100_weights
        self.weights_cifar100 = None
        self.create_new_model(base_model=self.model, first_trainable_layer=first_trainable_layer)

    def train(self, model, x_train, y_train, x_test, y_test):
        # training parameters
        batch_size = 64
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 25

        # The data, shuffled and split between train and test sets:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize_production(x_train), self.normalize_production(x_test)  # using production normalization

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                                       batch_size=batch_size),
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=maxepoches,
                                          validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        plot_figures(historytemp.history, "cifar10 trained on "+str(train_size)+" samples, regularization="+str(REG))
        model.save_weights('cifar10vgg_reg_{}_nSamples_{}.h5'.format(str(REG), str(train_size)))
        return model

    def create_new_model(self, base_model, first_trainable_layer):

        # idx of layer we want to change
        critical_layer_cifar100_idx = -6
        weights_cifar100_kernel = base_model.layers[critical_layer_cifar100_idx].get_weights()[0]  # 0=kernel (1=bias)
        weights_cifar100_bias = base_model.layers[critical_layer_cifar100_idx].get_weights()[1]

        def l2_reg_weight_diff_kernel(weights):
            return REG * K.sum(K.square(weights - weights_cifar100_kernel))

        def l2_reg_weight_diff_bias(weights):
            return REG * K.sum(K.square(weights - weights_cifar100_bias))

        if self.add_regularizer_for_cifar100_weights:

            # getting the configuration of the cifar100 model
            c = base_model.get_config()
            # changing the regularizer for the critical layer
            c['layers'][critical_layer_cifar100_idx]['config']['kernel_regularizer'] = l2_reg_weight_diff_kernel
            c['layers'][critical_layer_cifar100_idx]['config']['bias_regularizer'] = l2_reg_weight_diff_bias
            # creating a new base model (still suits cifar100) with the modified regularizer
            base_model = Sequential.from_config(c)
            # setting all weights as they were in the original cifar100 model
            base_model.model.load_weights('cifar100vgg.h5')

        # using base model to create our new model
        x = Dense(self.num_classes, activation='relu')(base_model.layers[-3].output)
        out = Activation('softmax')(x)

        new_model = Model(inputs=base_model.input, outputs=[out])
        new_model.summary()

        print("after")
        print("len", len(new_model.layers))

        # freeze layers
        for layer in new_model.layers[:first_trainable_layer]:
            layer.trainable = False

        self.model = new_model

        return new_model


class TransferEmbeddingsKNN:
    '''
    Folllows the guidelines in 3.2: Another approach for transfer learning in CNNs is by using the k-nearestneighbors
    (KNN) algorithm on embedding activations. A DNN embedding
    is typically considered as the second last layer in the network. This layer is
    known to represent semantic relatedness where semantically similar images embedded
    closer to each other. For transfer learning we take the training set of
    the new dataset (CIFAR-10) and map it to the embedding domain of CIFAR100.
    During inference, we map a new instance to this domain and run the KNN
    algorithm with the CIFAR-10 training set. Repeat this procedure for samples
    from CIFAR-10 of the sizes 100, 1000, 10000.
    In your report compare these two transfer learning methods (fine-tuning and
    KNN) and evaluate the results, try to motivate your result from a statistical
    learning theory perspective.
    '''
    def __init__(self, base_model):#, x_train, y_train, x_test, y_test):
        self.base_model = base_model
        self.model = self.remove_classification_head()
        self.model_knn = None

    def remove_classification_head(self):
        x = self.base_model.layers[-3].output
        new_model = Model(inputs=self.base_model.input, outputs=[x])
        return new_model

    def predict_embedding(self, x):
        embedding = self.model.predict(x)
        return embedding

    def train_knn(self, train_embeddings, train_y, n_neighbors=7):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(train_embeddings, train_y.ravel())
        self.model_knn = neigh
        return neigh

    def predict_knn(self, x):
        return self.model_knn.predict(x)

    def eval_knn(self,y_true, y_pred):
        # y_true = y_true.astype('float32')
        # y_true = K.constant(y_true)
        # y_pred = y_pred.astype('float32')
        # y_pred = K.constant(y_pred)
        # print(keras.losses.categorical_crossentropy(y_true, y_pred).loss)
        # print(keras.metrics.sparse_categorical_accuracy(y_true, y_pred).values)

        residuals = (y_pred.flatten() == y_true.flatten())
        acc = sum(residuals) / len(residuals)
        print("acc: ", acc)
        return acc


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    REG = 0

    # 3.1
    cifar_10_vgg = cifar10vgg()
    model = cifar_10_vgg.model

    for layer in model.layers:
        print(layer, layer.trainable)
    print(cifar_10_vgg.model.layers)

    for train_size in [100, 1000, 10000]:
        print("Training on size {}".format(train_size))
        X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
            X_train, y_train, train_size=train_size, random_state=42, stratify=y_train)

        model = try_loading_saved_weights(model, 'cifar10vgg_nSamples_{}.h5'.format(train_size))
        model = cifar_10_vgg.train(model, X_train_small, y_train_small, X_test_small, y_test_small)
        model.save_weights('cifar10vgg_nSamples_{}.h5'.format(train_size))
        predicted_x = model.predict(X_test)
        residuals = (np.argmax(predicted_x, 1) != np.argmax(y_test, 1))
        loss = sum(residuals) / len(residuals)
        print("the test 0/1 loss is: ", loss)

    # 3.2
    for train_size in [100, 1000, 10000]:
        history = {'acc': [], 'val_acc': []}
        neighbor_options = np.arange(1,45,4)
        for num_neighbors in neighbor_options:
            m = TransferEmbeddingsKNN(cifar100vgg(False).model)
            X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
                X_train, y_train, train_size=train_size, test_size=1000, random_state=42, stratify=y_train)
            train_embeddings = m.predict_embedding(X_train_small)
            test_embeddings = m.predict_embedding(X_test_small)
            m.train_knn(train_embeddings,y_train_small,n_neighbors=num_neighbors)
            acc = m.eval_knn(y_train_small, m.predict_knn(train_embeddings))
            preds = m.predict_knn(test_embeddings)
            val_acc = m.eval_knn(y_test_small, preds)
            history['acc'].append(acc)  # train acc
            history['val_acc'].append(val_acc)  # val acc
        plot_figures(history, "embeddings + knn for cifar10 trained on "+str(train_size)+" samples", metrics=['acc'],iterations=neighbor_options, x_axis_name='K (neighbors)')

    # 3.3
    for REG in {0, 0.5, 5}:
        REG = 0.5
        cifar_10_vgg = cifar10vgg(first_trainable_layer=-6, add_regularizer_for_cifar100_weights=True)
        model = cifar_10_vgg.model

        for layer in model.layers:
            print(layer, layer.trainable)
        print(cifar_10_vgg.model.layers)

        for train_size in [10000]:
            print("Training on size {}".format(train_size))
            X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
                X_train, y_train, train_size=train_size, random_state=42, stratify=y_train)

            model = try_loading_saved_weights(model, 'cifar10vgg_reg_{}_nSamples_{}.h5'.format(REG, train_size))
            model = cifar_10_vgg.train(model, X_train_small, y_train_small, X_test_small, y_test_small)

            predicted_x = model.predict(X_test)
            residuals = (np.argmax(predicted_x, 1) != np.argmax(y_test, 1))
            loss = sum(residuals) / len(residuals)
            print("the test 0/1 loss is: ", loss)
