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
import random

def maybe_download_data():
    # downloading data if necessary
    try:
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        print("downloading MNIST...")
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
        exps = np.exp(x - np.max(x))  # subtract max for numerical stability (as in course Q&A)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def backward(self, x, y):
        return y - x


# --------------
# Loss functions
# --------------
class mse:
    def forward(self, x, y):
        assert x.shape == y.shape
        return np.sum(0.5 * np.power(y - x, 2))

    def backward(self, x, y):
        assert x.shape == y.shape
        return -(y - x)


class cross_entropy:
    def forward(self, x, y):
        assert x.shape == y.shape
        log_likelihood = -np.log(x)
        # y is one-hot -> each sample's loss affected by a single element -> element-wise product and sum = sample loss.
        loss = np.sum(y * log_likelihood, axis=0)
        return loss

    def backward(self, x, y):
        assert x.shape == y.shape
        grad = y  # This is not the actual dC/dOut, but it is the portion that will backprop into softmax (dL/dz has a simple solution when the activation on z is softmax).
        return grad


# -----------
# Layer class
# -----------
class layer:
    """
    creates a fully-connected layer
    """

    def __init__(self, params):
        self._in_dim = params["input"]
        self._out_dim = params["output"]
        self._act_fn = params["nonlinear"]
        self._regularization = params["regularization"]
        self.name = ''

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
        self._grad_out = 0
        self._dw = np.zeros((self._out_dim, self._in_dim))
        self._db = np.zeros((self._out_dim, 1))

        self._grad_out = 0

    def forward(self, x, rglr=None):
        self._x = x
        self._z = np.dot(self._w, x) + self._b
        self._y = self._act_fn.forward(self._z)

        # For the first layer
        if rglr is None:
            rglr = 0

        return [self._y, self._regularization.forward(rglr + self._w)]

    def backward(self, grad_in):
        if self.name == 'cross-entropy':
            delta = self._act_fn.backward(grad_in, self._z)
        else:
            delta = grad_in * self._act_fn.backward(self._z)
        # grad_in is the incoming gradient from above
        self._dw = np.dot(delta, self._x.T)
        self._db = grad_in.dot(np.ones((delta.shape[1], 1)))
        self._grad_out = self._w.T.dot(delta)

    def reset(self):
        self._dw = 0
        self._db = 0
        self._grad_out = 0

    def update_grad(self, eta, w_decay=0.0):
        self._dw += w_decay * self._regularization.backward(self._w)
        self._w = self._w - eta * self._dw
        # print(self._w[0][0])
        self._b = self._b - eta * self._db

    def get_grad(self):
        return self._grad_out


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


def plot_figures(dict_x_y, title, metrics):
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

    for metric in metrics:
        for data_type in ['Train', 'Validation']:
            color = 'darkslategray' if 'Train' in data_type else 'c'
            metric_values = []
            num_backwards = []
            epoch_counter = 1
            # gather metric results from all epochs
            for epoch_dict in dict_x_y:
                metric_values.append(epoch_dict[data_type + ' ' + metric])
                num_backwards.append(epoch_counter * epoch_dict['n_backwards'])
                epoch_counter += 1
            # plot metric results
            if metric_values[0] is not None:
                df = pd.DataFrame.from_dict({'Steps': num_backwards, data_type + ' ' + metric: metric_values})
                plot_df(df, title, color)
        # display and close so next metric type is on new plot
        plt.savefig('../out/{}.png'.format(metric + ' for ' + title))
        plt.close()


def shuffle(x, y):
    indices = np.random.choice(range(y.shape[0]),len(y),replace=False)
    return x[indices], y[indices]


def to_one_hot(n_labels, data):
    return np.eye(n_labels)[data]


class mydnn:
    def __init__(self, architecture, loss, weight_decay=0.0):
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
        self.graph = self.build_graph()

        if self.loss == 'MSE':
            self.loss_func = mse()
        elif self.loss == 'cross-entropy':
            self.loss_func = cross_entropy()
        else:
            sys.exit("Error: undefined activation function {MSE, cross-entropy}.")

    def build_graph(self):
        layers = []
        for i in range(len(self.architecture)):
            layers.append(layer(self.architecture[i]))
        layers[-1].name = self.loss
        layers_reversed = layers.copy()
        layers_reversed.reverse()
        return layers, layers_reversed

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

        learning_rate = max(min_lr, learning_rate)
        learning_rate_original = learning_rate
        step_global = 0
        history = []

        # training
        for e in range(epochs):
            epoch_dict = {}
            start_time_epoch = time.time()
            x_train, y_train = shuffle(x_train, y_train)
            step_max = max(1, math.ceil(len(y_train) / batch_size))
            train_loss = 0
            train_acc = 0
            for step in range(step_max):
                batch_x = x_train[step * batch_size: (step + 1) * batch_size]
                batch_y = y_train[step * batch_size: (step + 1) * batch_size]

                batch_x = batch_x.T
                batch_y = batch_y.T

                # forward pass
                out, rglr = layers[0].forward(batch_x)
                for l in layers[1:]:
                    out, rglr = l.forward(out, rglr)
                loss = self.loss_func.forward(out, batch_y) + self.weight_decay*rglr
                train_loss += np.sum(loss)

                # accuracy  #TODO: only if cross-etropy
                accuracy = np.sum(np.argmax(out, 0) == np.argmax(batch_y, 0))
                train_acc += accuracy

                # backward pass
                grad = self.loss_func.backward(out, batch_y)
                for l in layers_reversed:
                    l.backward(grad)
                    grad = l.get_grad()

                # update gradients
                for l in layers_reversed:
                    l.update_grad(learning_rate / float(batch_size), self.weight_decay)

                step_global += 1

                # reset gradients and update learning rate for next round
                for l in layers_reversed:
                    l.reset()
                learning_rate = max(learning_rate_original * learning_rate_decay**(int(step_global/decay_rate)), min_lr)
                # print(learning_rate)

            train_loss = train_loss / x_train.shape[0]
            train_acc = 100 * train_acc / x_train.shape[0]

            val_loss, val_acc = self.evaluate(x_val, y_val, batch_size=batch_size)
            val_loss = np.mean(val_loss)

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
            batch_size = X.shape[1]

        step_max = max(1, math.ceil(X.shape[1]/ batch_size))
        for step in range(step_max):
            batch_x = X[:, step * batch_size: (step + 1) * batch_size]
            # first layer
            out, _ = layers_forward[0].forward(batch_x)
            # rest of layers
            for l in layers_forward[1:]:
                out, _ = l.forward(out)
            if len(pred) == 0:
                pred = out
            else:
                pred = np.append(pred, out, 1)
        return pred

    def evaluate(self, X, y, batch_size=None):
        """
        :param X: a 2d array, valid as an input to the network
        :param y: a 2d array, the labels of X in one-hot representation for classification or a value for each sample
        for regression.
        :param batch_size: an optional variable for splitting the prediction into batches for memory compliances.
        :return: [loss, accuracy] - for regression a list with the loss, for classification the
        loss and the accuracy
        """
        layers, layers_reversed = self.graph

        if batch_size is None:
            batch_size = X.shape[0]

        X, y = shuffle(X, y)
        step_max = max(1, math.ceil(len(y) / batch_size))
        train_loss, train_acc = 0, 0

        # Start evaluation
        for step in range(step_max):
            batch_x = X[step * batch_size: (step + 1) * batch_size]
            batch_y = y[step * batch_size: (step + 1) * batch_size]

            batch_x = batch_x.T
            batch_y = batch_y.T

            # forward pass
            out, rglr = layers[0].forward(batch_x)
            for l in layers[1:]:
                out, rglr = l.forward(out, rglr)
            loss = self.loss_func.forward(out, batch_y) + self.weight_decay*rglr
            train_loss += np.sum(loss)

            # accuracy
            pred = out
            accuracy = np.sum(np.argmax(pred, 0) == np.argmax(batch_y, 0))
            train_acc += accuracy

        train_loss = train_loss / X.shape[0]
        train_acc = 100 * train_acc / X.shape[0]

        return [train_loss, train_acc]


if __name__ == '__main__':
    # sys.stdout = open("d3_output_"+str(my_m)+".txt", "w")

    # MIST data preparations
    train_set, valid_set, test_set = maybe_download_data()
    train_set, valid_set, test_set = normalize(train_set, valid_set, test_set) # includes normalizing both train and validation according to train stats
    n_labels = len(np.unique(train_set[1]))
    x_train, y_train = train_set[0], train_set[1]
    x_val, y_val = valid_set[0], valid_set[1]
    x_test, y_test = test_set[0], test_set[1]
    y_train = to_one_hot(n_labels, y_train)
    y_val = to_one_hot(n_labels, y_val)
    y_test = to_one_hot(n_labels, y_test)

    '''
    Batch Size
    -------------
    We first consider a basic architecture with one hidden layer containing 128 neurons,
    a ReLU activation function, and softmax output layer. Your experiment
    should reveal the relationship between the batch size (with values 128, 1024 and
    10000) to the learning performance. Discuss your results, and design and run
    more experiments to support your hypothesis, if needed.
    '''
    _batch_size = [128, 1024, 10000]
    lr = 0.001
    epochs = 50
    for batch_size in _batch_size:
        print("Running batch size: {}".format(batch_size, lr))
        layer_1 = {"input": x_train.shape[1], "output": 128, "nonlinear": "relu", "regularization": "l1"}
        layer_2 = {"input": 128, "output": y_train.shape[1], "nonlinear": "softmax", "regularization": "l1"}
        model = mydnn(architecture=[layer_1, layer_2], loss="cross-entropy", weight_decay=0.0)
        history = model.fit(x_train, y_train, epochs, batch_size, lr, x_val=x_val, y_val=y_val)
        plot_figures(history, "Batch Size {}, Learning Rate: {}".format(batch_size, lr), metrics=['loss', 'accuracy'])


    '''
    Regularization
    -------------
    Consider the last (one hidden layer) architecture and run it first without regularization,
    and compare to applications with L1 and L2 norms regularization
    (optimize the weight decay parameter \lambda on the validation set; an initial recommended
    value is \lambda = 5e - 4). Discuss how the use of regularization affects
    generalization.
    '''
    batch_size = 128
    lr = 0.001
    _regularization = ["l1", "l1", "l2"]
    _wd = [0, 0.005, 0.005]
    epochs = 25
    i = 0
    for regularization in _regularization:
        print("Running regularization: {}, wd: {}".format(regularization, _wd[i]))
        layer_1 = {"input": x_train.shape[1], "output": 128, "nonlinear": "relu", "regularization": regularization}
        layer_2 = {"input": 128, "output": y_train.shape[1], "nonlinear": "softmax", "regularization": regularization}
        model = mydnn(architecture=[layer_1, layer_2], loss="cross-entropy", weight_decay=_wd[i])
        history = model.fit(x_train, y_train, epochs, batch_size, lr, x_val=x_val, y_val=y_val)
        plot_figures(history, "Regularization: {}, wd: {}".format(regularization, _wd[i]), metrics=['loss', 'accuracy'])

        # Plot weights histogram
        plt.hist(model.graph[1][0]._w.flatten(), 20)
        plt.xlabel('Value')
        plt.ylabel('#')
        plt.title('Weights Distribution, regularization: {}, wd: {}'.format(regularization, _wd[i]))
        plt.show()

        i += 1

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
    def make_architecture(depth, width, in_shape, out_shape, activ, rglr, loss):
        print("depth {}, width {}".format(depth, width))
        output_shape = width
        architecture = []
        layer_ids = range(depth)
        for layer_id in layer_ids:
            layer_dict = {}
            # input layer handled differently:
            if layer_id == 0:
                layer_dict["input"] = in_shape
            else:
                layer_dict["input"] = output_shape
            # output layer handled differently:
            if layer_id == layer_ids[-1]:
                layer_dict["output"] = out_shape
                if loss == 'cross-entropy':
                    layer_dict["nonlinear"] = "softmax"
                else:
                    layer_dict["nonlinear"] = activ
            else:
                layer_dict["output"] = width
                layer_dict["nonlinear"] = activ
            layer_dict["regularization"] = rglr
            architecture.append(layer_dict)
        return architecture


    depth_options, width_options = np.arange(1,4), np.arange(1, 513, 100)
    best_acc = 0
    rglr = "l1"
    activ = "relu"
    loss = "cross-entropy"
    weight_decay = 0.0
    for lr in [0.007]:
        try:
            for depth in depth_options:
                for width in width_options:
                    # if depth != 3 or width != 501:
                    #     continue
                    is_best = False
                    architecture = make_architecture(depth, width, x_train.shape[1], y_train.shape[1], activ, rglr, loss)

                    model = mydnn(architecture=architecture, loss=loss, weight_decay=weight_decay)
                    history = model.fit(x_train, y_train, 70, 64, lr, 1, 1000, x_val=x_val, y_val=y_val)
                    last_epoch_acc = history[-1]["Validation accuracy"]
                    if last_epoch_acc > best_acc:  # This is only a single measure. The loss and accuracy curves are more informative to determine best model.
                        is_best = True
                        best_acc = last_epoch_acc
                        best_architecture = architecture
                    plot_figures(history, "depth {}   width: {}   (lr: {})"
                                 .format(str(depth), str(width), str(lr)), metrics=['loss', 'accuracy'])
                    print(model.evaluate(x_test, y_test, 64))
        except:
            continue


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

    def plot_3d(xy_mesh, z, title):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # data prep
        X, Y = xy_mesh
        reshape_size = int(np.sqrt(z.shape[1]))
        Z = z.reshape((reshape_size, reshape_size))

        # plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # setting axes
        ax.set_zlim(-0.3, 0.3)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # color bar and saving
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('../out/3d_'+title+'.png')
        plt.close()
        # plt.show()

    def f(x):
        x1, x2 = x[0], x[1]
        return [x1 * np.exp(-x1 ** 2 - x2 ** 2)]

    # test set
    test_x_mesh = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))  # We later use a test set with 1000
    test_x = np.array([test_x_mesh[0].flatten(), test_x_mesh[1].flatten()]).T
    test_y = np.apply_along_axis(f, 1, test_x)

    # train set
    for m in [100, 1000]:
        train_x, train_y = [], []
        train_x = np.random.uniform(low=-2, high=2, size=(m, 2))
        train_y = np.apply_along_axis(f, 1, train_x)

        depth_options, width_options = np.arange(1, 4), np.arange(1, 513, 100)
        best_acc = 0
        rglr = "l1"
        activ = "sigmoid"
        loss = "MSE"
        weight_decay = 0.0
        batch_size = 100
        for lr in [0.007]:
            print("lr", lr)
            for depth in [2,3]:
                for width in [2**i for i in range(4,9)]:
                    is_best = False
                    architecture = make_architecture(depth, width, train_x.shape[1], train_y.shape[1], activ, rglr, loss)

                    model = mydnn(architecture=architecture, loss=loss, weight_decay=weight_decay)
                    history = model.fit(train_x, train_y, 150000, batch_size, lr, 1, 1000*float(m/batch_size), x_val=test_x, y_val=test_y)
                    last_epoch_acc = history[-1]["Validation loss"]
                    if last_epoch_acc > best_acc:
                        is_best = True
                        best_acc = last_epoch_acc
                        best_architecture = architecture
                    plot_figures(history, "short_test_Q4 is_best {}  n_samples: {}  depth: {}   width: {}   (lr: {})"
                                 .format(str(is_best), str(m), str(depth), str(width), str(lr)), metrics=['loss'])
                    # test set with 1000 points in linspace per x and y
                    test_x_mesh = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
                    test_x = np.array([test_x_mesh[0].flatten(), test_x_mesh[1].flatten()]).T
                    test_y = np.apply_along_axis(f, 1, test_x)
                    preds = model.predict(test_x.T, 100)
                    plot_3d(test_x_mesh, preds, "short_test_is_best {}  n_samples: {}  depth: {}   width: {}   (lr: {})"
                            .format(str(is_best), str(m), str(depth), str(width), str(lr)))