# Submitted by:
# (1) Gil Shomron (301722294)
# (2) Alona Levy (300872025)

import pickle, gzip, urllib.request, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def download_data():
    # downloading data if necessary
    try:
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"  # Load the dataset
        urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    return train_set, valid_set, test_set


class layer:
    # TODO: A   (this is a rough draft based on tutorial 4, currently only forward pass for scalars is working)
    """
    creates a fully-connected layer
    """

    def __init__(self, W, x, b):
        self.W = self.variable(W)
        self.x = self.variable(x)
        self.b = self.variable(b)
        self.value = self.forward()
        self.gradient = 0
        self.gradient_sum = 0

    def forward(self):
        return self.add(self.mult(self.W, self.x), self.b).forward()

    def backward(self):
        return 0

    def reset(self):
        return 0

    class variable:
        def __init__(self, value):
            self.value = value
            self.gradient = 0
            self.gradient_sum = 0

        def forward(self):
            return self.value

        def set_value(self, value):
            self.value = value

        def backward(self, grad=None):
            if grad is None:
                self.gradient = 1
            else:
                self.gradient_sum += grad
                self.gradient += grad

        def reset(self):
            self.gradient = 0

        def update_grad(self, eta):
            self.value = self.value - 1 * eta * self.gradient_sum
            self.gradient_sum = 0

        def get_gradient(self):
            return self.gradient

    class add:
        def __init__(self, child, other):
            self.child = child
            self.other = other
            self.value = 0
            self.gradient = 0

        def forward(self):
            self.value = self.child.forward() + self.other.forward()
            return self.value

        def backward(self, grad=None):
            if grad is None:
                self.gradient = 1
            else:
                self.gradient += grad

            self.child.backward(self.gradient * 1)
            self.other.backward(self.gradient * 1)

        def reset(self):
            self.gradient = 0
            self.child.reset()
            self.other.reset()

    class mult:
        def __init__(self, child, other):
            self.child = child
            self.other = other
            self.value = 0
            self.gradient = 0

        def forward(self):
            self.value = self.child.forward() * self.other.forward()
            return self.value

        def backward(self, grad=None):
            if grad is None:
                self.gradient = 1
            else:
                self.gradient += grad

            self.child.backward(self.gradient * self.other.value)
            self.other.backward(self.gradient * self.child.value)

        def reset(self):
            self.gradient = 0
            self.child.reset()
            self.other.reset()


class activations:
    # TODO A
    """
    Need all activations (including softmax, RELU etc. - see in HW)
    """


class loss:
    # TODO A
    """
    loss, accuracy and whatever they asked to measure
    """


class regularization:
    # TODO A
    """
    any regularization functions required
    """


class mydnn:

    # TODO B
    def __init__(self, architecture, loss, weight_decay=0):
        """
        :param architecture: A list of dictionaries, each dictionary represents a layer, for each layer the dictionary
         will consist
            – “input” - int, the dimension of the input
            – “output” int, the dimension of the output
            – “nonlinear” string, whose possible values are: “relu”, “sigmoid”, “sotmax” or “none”
            – “regularization” string, whose possible values are: “l1” (L1 norm), or “l2” (L2 norm)
        :param loss: string, could be one of “MSE” or “cross-entropy”
        :param weight_decay: float, the lambda parameter for the regularization.
        """

    def _normalize(self, x_train, x_val, x_test):
        # subtract training set's mean sample from all samples
        sample_mean = x_train.mean(axis=0, keepdims=True)
        x_train = x_train - sample_mean
        x_val = x_val - sample_mean
        x_test = x_test - sample_mean

        return x_train, x_val, x_test

    def _plot_figures(self, dict_x_y, title):
        """
        the results should be depicted in two figures (one for loss
        and one for accuracy), where the X-axis is the number of iterations (number of
        backward iterations during training) and the Y -axis will be the loss or accuracy.
        In each figure show both the training and validation curves.
        :param dict_x_y - a dictionary of the form {'x_axis_name': [], 'y_axis_name': []} corresponding to
                the x and y values to plot. E.g.: {'Steps': [1,2,..5], 'Accuracy': [89, 95,..., 100]}
        :param title - plot title
        """

        # preparing data for easy plotting
        df = pd.DataFrame.from_dict(dict_x_y)
        x_name = df.columns[0]
        y_name = df.columns[1]

        # plotting
        plt.plot(x_name, y_name, data=df, marker='', color='green', linewidth=2)
        # plt.plot('x', 'train_sq', data=df, marker='', color='blue', linewidth=2)
        # plt.plot('x', 'test_01', data=df, marker='', color='orange', linewidth=2)  # linestyle='dashed', label="toto")
        # plt.plot('x', 'test_sq', data=df, marker='', color='red', linewidth=2)  # linestyle='dashed', label="toto")
        plt.legend()

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(title)
        plt.show()

    # TODO B
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0,
            decay_rate=1, min_lr=0.0, x_val=None, y_val=None, ):
        """
        The function will run SGD for a user-defined number of epochs, with the
        defined batch size. On every epoch, the data will be reshuffled (make sure you
        shuffle the x’s and y’s together).
        For every batch the data should be passed forward and gradients pass backward.
        After gradients are computed, weights update will be performed using
        the current learning rate.
        The learning rate for time step t is computed by:
        lr (t) = max(learning_rate * learning_rate_decay^(t/decay_rate), min_lr)
        After every epoch the following line will be printed on screen with the values
        of the last epoch:
        Epoch 3/5 - 30 seconds - loss: 0.3 - acc: 0.9 - val_loss: 0.4 - val_acc: 0.85
        (The accuracy will be printed only for classification networks, i.e., networks
        which cross entropy is their final loss.
        :param x_train: a Numpy nd-array where each row is a sample
        :param y_train: a 2d array, the labels of X in one-hot representation for classification or a value for each
         sample for regression
        :param epochs: number of epochs to run
        :param batch_size: batch size for SGD
        :param learning_rate: float, an initial learning rate that will be used in SGD.
        :param learning_rate_decay: a factor which will multiply the learning rate every decay_rate minibatches.
        :param decay_rate:  the number of steps between every two consecutive learning rate decays (one forward+backward
         of a single minibatch count as a single step).
        :param min_lr: a lower bound on the learning rate.
        :param x_val: the validation x data (same structure as train data) default is None. When validation data is
        given, evaluation over this data will be made at the end of every epoch.
        :param y_val: the corresponding validation y data (labels) whose structure is identical to y_train.

        :return: history - intermediate optimization results, which is a list of dictionaries, such that each epoch has
         a corresponding dictionary containing all relevant results. These dictionaries do not contain formatting
         information (you will later use the history to print various things including plots of learning and convergence
          curves for your networks). The exact structure of each dictionary is up to you.
        """
        history = None
        return history

    def predict(self, X, batch_size=None):
        # TODO B
        """
        The predict function will get an nd-array of inputs and return the network prediction
        :param X:  a 2d array, with valid dimensions for the network.
        :param batch_size:  an optional variable for splitting the prediction into batches for memory compliances; if
        None, all the samples will be processed in a single batch
        :return:  pred - a 2d array where each row is a prediction of the corespondent sample
        """
        pred = None
        return pred

    def evaluate(self, X, y, batch_size=None):
        # TODO A
        """

        :param X: a 2d array, valid as an input to the network
        :param y: a 2d array, the labels of X in one-hot representation for classification or a value for each sample
        for regression.
        :param batch_size: an optional variable for splitting the prediction into batches for memory compliances.
        :return: [loss, accuracy] - for regression a list with the loss, for classification the
        loss and the accuracy
        """
        [loss, accuracy] = None
        return [loss, accuracy]


if __name__ == '__main__':
    train_set, valid_set, test_set = download_data()

    # TODO: A
    '''
    Batch Size 
    -------------
    We first consider a basic architecture with one hidden layer containing 128 neurons,
    a ReLU activation function, and softmax output layer. Your experiment
    should reveal the relationship between the batch size (with values 128, 1024 and
    10000) to the learning performance. Discuss your results, and design and run
    more experiments to support your hypothesis, if needed.
    '''
    model = mydnn(architecture=None, loss=None)
    print(layer(1,2,3).forward())
    model._plot_figures({'Steps':[1,2,3], 'Accuracy': [98,100,89]}, 'Test')
    model.fit(...)

    # TODO: A
    '''
    Regularization
    -------------
    Consider the last (one hidden layer) architecture and run it first without regularization,
    and compare to applications with L1 and L2 norms regularization
    (optimize the weight decay parameter λ on the validation set; an initial recommended
    value is λ = 5e − 4). Discuss how the use of regularization affects
    generalization.
    '''
    model = mydnn(architecture=None, loss=None)
    model.fit(...)
    model._plot_figures(...)

    # TODO: B
    '''
    Architecture:
    -------------
    Architecture selection is major and challenging task when constructing DNNs.
    Research how the architecture selection affects the results for the MNIST dataset.
    Find the best architecture from all architectures up to depth of 3 and width of
    512 (conduct a grid search of at least 9 different architectures). In your report
    discuss how the width, depth and the overhaul number of weights affect the
    train and test accuracy. Motivate your selection of architecture from a statistical
    learning theory perspective (hypothesis set size, training set size, overfitting
    etc...).
    '''
    model = mydnn(architecture=None, loss=None)
    model.fit(...)
    model._plot_figures(...)

    # TODO: B
    '''
    Regression:
    -------------
    For this section we will create a synthetic dataset using the function
    f(x) = x_1 exp(−x_1^2-x_2^2)
    Sample uniformly at random m training points in the range x1 ∈ [−2, 2], x2 ∈
    [−2, 2]. For the test set, take the linear grid using np.linspace (-2,2, 1000).
    Find the best architecture for the case where m = 100 and for the case
    m = 1000. In your results show the final MSE on the test set and also plot
    a 3d graph showing yˆtest (predicted values for the test points) as function of
    x = (x1, x2).
    '''
    model = mydnn(architecture=None, loss=None)
    model.fit(...)
    model._plot_figures(...)
    model._plot_3d_figure(...)