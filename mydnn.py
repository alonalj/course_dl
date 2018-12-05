# Submitted by:
# (1) Gil Shomron (301722294)
# (2) Alona Levy (300872025)

import pickle, gzip, urllib.request, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb
import math
import time


def maybe_download_data():
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


# --------------
# Regularization
# --------------
class l1_norm:
    def forward(self, w):
        return np.linalg.norm(w, ord=1)

    def backward(self, w):
        mask1 = (w >= 0) * 1.0
        mask2 = (w < 0) * -1.0
        return mask1 + mask2


class l2_norm:
    def forward(self, w):
        return np.linalg.norm(w, ord=2)

    def backward(self, w):
        return 2 * w


# --------------------
# Activation functions
# --------------------
class relu:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0) * 1


class sigmoid:
    def forward(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def backward(self, x):
        return np.multiply(self.forward(x), 1 - self.forward(x))


class softmax:
    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, x):
        return x


# --------------
# Loss functions
# --------------
class mse:
    def forward(self, x, y):
        assert x.shape == y.shape
        return np.sum(np.mean(np.square(y - x), axis=1)) # TODO: fixed to mean over samples and sum over classes, although they mentioned we can assume that for classification we will use only cross-entropy

    def backward(self, x, y):
        assert x.shape == y.shape
        return -2 * (y - x)


# TODO: fix this
class cross_entropy:
    def forward(self, x, y):
        assert x.shape == y.shape
        s = softmax()

        num_examples = y.shape[0]
        log_likelihood = -np.log(x[range(num_examples), y])
        loss = np.sum(log_likelihood) / num_examples
        return loss

    def backward(self, x, y):
        assert x.shape == y.shape
        s = softmax()

        num_examples = y.shape[0]
        grad = x
        grad[range(num_examples), y] -= 1
        grad = grad/num_examples
        return grad


# -----------
# Layer class
# -----------
class layer:
    """
    creates a fully-connected layer
    """

    def __init__(self, params):
        # TODO: add assertions
        self._in_dim = params["input"]
        self._out_dim = params["output"]
        self._act_fn = params["nonlinear"]
        self._regularization = params["regularization"]

        # replace strings with function pointer
        if self._act_fn == "relu":
            self._act_fn = relu()
        elif self._act_fn == "sigmoid":
            self._act_fn = sigmoid()
        elif self._act_fn == "softmax":
            self._act_fn = softmax()
        else:
            sys.exit("Error: undefined activation function {relu, sigmoid, softmax}.")

        if self._regularization == "l1":
            self._regularization = l1_norm()
        elif self._regularization == "l2":
            self._regularization = l2_norm()
        else:
            sys.exit("Error: undefined activation function {l1, l2}.")

        self._w = np.random.uniform(-1 / np.sqrt(self._in_dim),
                                     1 / np.sqrt(self._in_dim),
                                    (self._out_dim, self._in_dim))
        # parameters
        self._b = np.zeros((self._out_dim, 1))
        self._x = 0
        self._y = 0
        self._z = 0

        # parameter gradients
        self._delta = 0
        self._dw = np.zeros((self._out_dim, self._in_dim))
        self._db = np.zeros((self._out_dim, 1))

        self._grad = 0

    def forward(self, x, rglr=None):
        self._x = x
        self._z = np.matmul(self._w, x) + self._b
        self._y = self._act_fn.forward(self._z)

        # For the first layer
        if rglr is None:
            rglr = 0

        return [self._y, self._regularization.forward(rglr + self._w)]

    def backward(self, grad):
        batch_size = self._x.shape[1]

        self._delta = np.multiply(grad, self._act_fn.backward(self._z))
        self._dw += (1/batch_size) * np.matmul(self._delta, self._x.T)
        self._db += np.expand_dims(np.sum(self._delta, axis=1), axis=1)
        self._grad += np.matmul(self._w.T, self._delta)

    def reset(self):
        self._dw = 0
        self._db = 0
        self._grad = 0

    def update_grad(self, eta, w_decay=0):
        self._dw += w_decay * self._regularization.backward(self._w)
        self._w = self._w - eta * self._dw
        self._b = self._b - eta * self._db

    def get_grad(self):
        return self._grad


# Debug
params1 = {"input": 2,
           "output": 4,
           "nonlinear": "relu",
           "regularization": "l1"}

params2 = {"input": 4,
           "output": 1,
           "nonlinear": "relu",
           "regularization": "l1"}


def normalize(train, val, test):
    # subtract training set's mean sample from all samples
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train, val, test
    sample_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - sample_mean
    x_val = x_val - sample_mean
    x_test = x_test - sample_mean

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def maybe_expand_dims(data):
    if len(data.shape) != 2:
        data = np.expand_dims(data, 1)
    return data


def plot_figures(dict_x_y, title):
    """
    the results should be depicted in two figures (one for loss
    and one for accuracy), where the X-axis is the number of iterations (number of
    backward iterations during training) and the Y -axis will be the loss or accuracy.
    In each figure show both the training and validation curves.
    :param dict_x_y - a dictionary of the form {'x_axis_name': [], 'y_axis_name': []} corresponding to
            the x and y values to plot. E.g.: {'Steps': [1,2,..5], 'Accuracy': [89, 95,..., 100]}
    :param title - plot title
    """

    def plot_df(df, title, color):
        # plot using a data frame
        x_name = df.columns[0]
        y_name = df.columns[1]

        plt.plot(x_name, y_name, data=df, marker='', color=color, linewidth=2)
        plt.legend()
        plt.xlabel(x_name)
        plt.ylabel(y_name.split(' ')[1].title())
        plt.title(title)

    for metric in ['loss', 'accuracy']:
        for data_type in ['Train', 'Validation']:
            color = 'darkslategray' if 'Train' in data_type else 'c'
            metric_values = []
            num_backwards = []
            epoch_counter = 1
            # gather metric results from all epochs
            for epoch_dict in dict_x_y:
                metric_values.append(epoch_dict[data_type + ' ' + metric])
                num_backwards.append(epoch_counter*epoch_dict['n_backwards'])
                epoch_counter += 1
            # plot metric results
            if metric_values[0] is not None:
                df = pd.DataFrame.from_dict({'Steps':num_backwards, data_type + ' ' + metric: metric_values})
                plot_df(df, title, color)
        # display and close so next metric type is on new plot
        plt.show()  # TODO: replace with plt.save?
        plt.close()


def shuffle(x, y):
    indices = np.random.choice(range(y.shape[0]),len(y),replace=False)
    return x[indices], y[indices]


class mydnn:
    # TODO B
    def __init__(self, architecture, loss, weight_decay=0):
        """
        :param architecture: A list of dictionaries, each dictionary represents a layer, for each layer the dictionary
         will consist
            - "input" - int, the dimension of the input
            - "output" int, the dimension of the output
            - "nonlinear" string, whose possible values are: "relu", "sigmoid", "sotmax" or "none"
            - "regularization" string, whose possible values are: "l1" (L1 norm), or "l2" (L2 norm)
        :param loss: string, could be one of "MSE" or "cross-entropy"
        :param weight_decay: float, the lambda parameter for the regularization.
        """
        self.architecture = architecture
        self.loss = loss
        self.weight_decay = weight_decay
        # self.debug = debug  # TODO: remove from signature as well as from here
        self.graph = self.build_graph()

    def build_graph(self):
        layers = []
        for i in range(len(self.architecture)):
            layers.append(layer(self.architecture[i]))
        layers_reversed = layers.copy()
        layers_reversed.reverse()
        return layers, layers_reversed

    # TODO B
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0,
            decay_rate=1, min_lr=0.0, x_val=None, y_val=None, ):
        """
        The function will run SGD for a user-defined number of epochs, with the
        defined batch size. On every epoch, the data will be reshuffled (make sure you
        shuffle the x's and y's together).
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
        layers, layers_reversed = self.graph

        # loss function
        if self.loss == 'MSE':
            loss_func = mse()
            metric_name = 'Loss'
        elif self.loss == 'cross-entropy':
            loss_func = cross_entropy()
            metric_name = 'Accuracy'
        else:
            sys.exit("Error: undefined activation function {MSE, cross-entropy}.")

        learning_rate = max(min_lr, learning_rate)
        step_counter_tot = 0
        history = []

        # training
        for e in range(epochs):
            epoch_dict = {}
            start_time_epoch = time.time()
            x_train, y_train = shuffle(x_train, y_train)
            step_max = max(1, math.ceil(len(y_train) / batch_size))
            for step in range(step_max):  # TODO verify this doesn't skip any batches
                batch_x = x_train[step * batch_size: (step + 1) * batch_size]
                batch_y = y_train[step * batch_size: (step + 1) * batch_size]

                # expand batch dimensions if necessary:
                batch_x = maybe_expand_dims(batch_x)
                batch_y = maybe_expand_dims(batch_y)

                # forward pass
                out, rglr = layers[0].forward(batch_x.T)
                for l in layers[1:]:
                    out, rglr = l.forward(out, rglr)
                loss = loss_func.forward(out + self.weight_decay*rglr, batch_y.T)  # the average loss over batch

                # backward pass
                grad = loss_func.backward(out, batch_y.T)
                for l in layers_reversed:
                    l.backward(grad)
                    grad = l.get_grad()

                # update gradients
                for l in layers_reversed:
                    l.update_grad(learning_rate / batch_size, self.weight_decay)

                # save and print results
                if step % 100 == 0:
                    print("iteration {}/{} - loss {}".format(step, step_max, loss.T))
                step_counter_tot += 1

                # reset gradients and update learning rate for next round
                for l in layers_reversed:
                    l.reset()
                learning_rate = max(learning_rate * learning_rate_decay**(int(step/decay_rate)), min_lr)

            train_loss, train_acc = self.evaluate(x_train.T, y_train.T)
            val_loss, val_acc = self.evaluate(x_val.T, y_val.T)

            # saving to epoch dictionary
            epoch_dict['Train accuracy'] = train_acc
            epoch_dict['Validation accuracy'] = val_acc
            epoch_dict['Train loss'] = train_loss
            epoch_dict['Validation loss'] = val_loss
            epoch_dict['n_backwards'] = step_max

            # printing
            duration_epoch = time.time() - start_time_epoch
            if self.loss == 'MSE':
                print("Epoch {}/{} - {} seconds - loss: {} - val_loss: {} "
                      .format(e, epochs, duration_epoch, train_loss, val_loss))
            else:
                print("Epoch {}/{} - {} seconds - loss: {} - acc: {} - val_loss: {} - val_acc: {}"
                      .format(e, epochs, duration_epoch, train_loss, train_acc, val_loss, val_acc))

            history.append(epoch_dict)

        return history

    def predict(self, X, batch_size=None):
        """
        The predict function will get an nd-array of inputs and return the network prediction
        :param X:  a 2d array, with valid dimensions for the network.
        :param batch_size:  an optional variable for splitting the prediction into batches for memory compliances; if
        None, all the samples will be processed in a single batch
        :return:  pred - a 2d array where each row is a prediction of the corespondent sample
        """
        pred = []
        layers_forward, _ = self.graph
        if batch_size is None:
            batch_size = len(X)

        step_max = max(1, math.ceil(len(X) / batch_size))
        for step in range(step_max):
            batch_x = X[step * batch_size: (step + 1) * batch_size]
            # first layer
            out, _ = layers_forward[0].forward(batch_x)
            # rest of layers
            for l in layers_forward[1:]:
                out, _ = l.forward(out)
            pred.extend(out)
        return np.array(pred)

    def evaluate(self, X, y, batch_size=None):
        # TODO A - should be complete
        """
        :param X: a 2d array, valid as an input to the network
        :param y: a 2d array, the labels of X in one-hot representation for classification or a value for each sample
        for regression.
        :param batch_size: an optional variable for splitting the prediction into batches for memory compliances.
        :return: [loss, accuracy] - for regression a list with the loss, for classification the
        loss and the accuracy
        """
        pred = self.predict(X, batch_size)
        accuracy = None
        if self.loss == 'MSE':
            loss_func = mse()
        else:
            loss_func = cross_entropy()
            accuracy = 100 * np.sum(np.argmax(pred, 1) == y) / float(len(y))
        loss = loss_func.forward(pred, y)  # loss functions already handle averaging over batch
        return [loss, accuracy]


if __name__ == '__main__':

    # MIST data preparations
    train_set, valid_set, test_set = maybe_download_data()
    train_set, valid_set, test_set = normalize(train_set, valid_set, test_set) # includes normalizing both train and validation according to train stats
    x_train, y_train = maybe_expand_dims(train_set[0]), maybe_expand_dims(train_set[1])
    x_val, y_val = maybe_expand_dims(valid_set[0]), maybe_expand_dims(valid_set[1])
    x_test, y_test = maybe_expand_dims(test_set[0]), maybe_expand_dims(test_set[1])

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
    # model = mydnn(architecture=None, loss=None)
    # print(layer(1,2,3).forward())
    # model._plot_figures({'Steps':[1,2,3], 'Accuracy': [98,100,89]}, 'Test')
    # model.fit(...)

    # TODO: A
    '''
    Regularization
    -------------
    Consider the last (one hidden layer) architecture and run it first without regularization,
    and compare to applications with L1 and L2 norms regularization
    (optimize the weight decay parameter \lambda on the validation set; an initial recommended
    value is \lambda = 5e - 4). Discuss how the use of regularization affects
    generalization.
    '''
    # model = mydnn(architecture=None, loss=None)
    # model.fit(...)
    # model._plot_figures(...)

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
    depth_options, width_options = np.arange(1,4), np.arange(1, 513)
    for depth in depth_options:
        # print("depth", depth)
        for num_neurons in width_options:
            output_shape = num_neurons
            architecture = []
            layer_ids = range(depth)
            for layer_id in layer_ids:
                layer_dict = {}
                if layer_id == 0:
                    layer_dict["input"] = x_train.shape[1]
                else:
                    layer_dict["input"] = output_shape

                if layer_id == layer_ids[-1]:  # last layer
                    layer_dict["output"] = y_train.shape[1]
                else:
                    layer_dict["output"] = num_neurons

                if layer_id == layer_ids[-1]: # last layer with softmax
                    layer_dict["nonlinear"] = "softmax"
                else:
                    layer_dict["nonlinear"] = "relu"

                layer_dict["regularization"] = "l1"
                architecture.append(layer_dict)
            output_shape = layer_dict["output"]
            model = mydnn(architecture=architecture, loss="cross-entropy")
            history = model.fit(x_train, y_train, 20, 125, 0.001, 0.99, 1000, x_val=x_val, y_val=y_val)
            plot_figures(history, "test")

    # TODO: B
    '''
    Regression:
    -------------
    For this section we will create a synthetic dataset using the function
    f(x) = x_1 exp(-x_1^2-x_2^2)
    Sample uniformly at random m training points in the range x1 in [-2, 2], x2 in
    [-2, 2]. For the test set, take the linear grid using np.linspace (-2,2, 1000).
    Find the best architecture for the case where m = 100 and for the case
    m = 1000. In your results show the final MSE on the test set and also plot
    a 3d graph showing y'test (predicted values for the test points) as function of
    x = (x1, x2).
    '''
    # model = mydnn(architecture=None, loss=None)
    # model.fit(...)
    # model._plot_figures(...)
    # model._plot_3d_figure(...)



    # -----------------------
    # TESTING - INTERNAL ONLY #TODO: REMOVE
    # -----------------------

    # ONE LAYER
    print("Testing one-layer")
    for batch_size in [1]:
        print("Batch size", batch_size)
        model = mydnn(architecture=[{"input": 2, "output": 1, "nonlinear": "relu", "regularization": "l1"}], loss="MSE",
                      )
        model.fit(np.array([[4, 0], [0, 1]]), np.array([[4], [1]]), 10, batch_size, 0.01)  # classification

    # TWO LAYER
    print("Testing two-layer")
    for batch_size in [1, 2]:
        print("Batch size", batch_size)
        model = mydnn(architecture=[{"input": 2, "output": 2, "nonlinear": "relu", "regularization": "l1"},
                                    {"input": 2, "output": 1, "nonlinear": "relu", "regularization": "l1"}], loss="MSE",
                      )
        model.fit(np.array([[1, 0], [0, 1]]), np.array([[1], [0]]), 10, batch_size, 0.001)  # classification

    # TWO LAYER, WIDE HIDDEN LAYER
    print("Testing two-layer, wide hidden layer")
    for batch_size in [16]:
        print("Batch size", batch_size)
        model = mydnn(architecture=[{"input": 2, "output": 128, "nonlinear": "relu", "regularization": "l1"},
                                    {"input": 128, "output": 2, "nonlinear": "relu", "regularization": "l1"}], loss="MSE")
        model.fit(np.array([[4, 1], [0, 1], [3, 9], [3, 3]]),
                  np.array([[5, 3], [1, -1], [12, -6], [6, 0]]), 10, batch_size, 0.01)  # classification





