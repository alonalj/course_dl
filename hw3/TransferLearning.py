from keras.layers import Dense, Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras import Model
from cifar100vgg import *
# TODO verify the above import works consistently


def plot_figures(dict_x_y, title, metrics=['loss', 'acc']):
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
            num_epochs = range(len(dict_x_y[data_type + metric]))

            # plot metric results
            if metric_values[0] is not None:
                df = pd.DataFrame.from_dict({'Epochs': num_epochs, data_type_name + ' ' + metric: metric_values})
                plot_df(df, title, color)
        # display and close so next metric type is on new plot
        plt.savefig('../../out/{}.png'.format(metric + ' for ' + title))
        # plt.show()
        plt.close()


class cifar10vgg(cifar100vgg):

    def __init__(self, train=False):
        super().__init__(train)
        self.num_classes = 10
        self.create_new_model(base_model=self.model)

    def train(self, model, x_train, y_train, x_test, y_test):
        # training parameters
        batch_size = 64
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

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
        plot_figures(historytemp.history, "cifar10 trained on "+str(train_size)+" samples")
        model.save_weights('cifar10vgg.h5')
        return model

    def create_new_model(self, base_model):
        # Check the trainable status of the individual layers
        print("before")
        print("len", len(base_model.layers))

        x = Dense(self.num_classes, activation='relu')(base_model.layers[-3].output)
        out = Activation('softmax')(x)

        new_model = Model(input=base_model.input, output=[out])
        new_model.summary()

        for layer in new_model.layers[:-2]:
            print(layer, layer.trainable)

        print("after")
        print("len", len(new_model.layers))

        # freeze layers
        for layer in new_model.layers[:-2]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in new_model.layers:
            print(layer, layer.trainable)

        self.model = new_model

        return new_model


'''
replace the last fully-connected layer with a new
initialized layer, where the output size is the number of classes in CIFAR-10 (i.e.,
10 classes). We will then freeze all other layers and train the last layer only with
the small (sample) available dataset for CIFAR-10. Repeat this procedure for
datasets sampled from CIFAR-10 of sizes 100, 1000, 10000. Make sure you use
the random seed 42 so we can reproduce your results.
'''

if __name__ == '__main__':
    cifar_10_vgg = cifar10vgg()
    model = cifar_10_vgg.model
    print(cifar_10_vgg.model.layers)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    for train_size in [100, 1000, 10000]:
        print("Training on size {}".format(train_size))
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42,
                                                              stratify=y_train)

        model = cifar_10_vgg.train(model, X_train_small, y_train_small, X_test, y_test)

        predicted_x = model.predict(X_test)
        residuals = (np.argmax(predicted_x, 1) != np.argmax(y_test, 1))
        loss = sum(residuals) / len(residuals)
        print("the validation 0/1 loss is: ", loss)
