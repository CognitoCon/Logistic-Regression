import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta, X, y, lam=0.0):
    m = len(y)
    h = sigmoid(X @ theta)

    J = (1 / m) * ((-y).T @ np.log(h) - (1 - y).T @ np.log(1 - h)) + (
        lam / (2 * m)
    ) * theta.T @ theta

    grad = (h - y).T @ X + (lam / m) * theta.T

    return J, grad


def gradientDescent(X, y, theta, alpha, num_iters, lam=0):

    J_history = np.zeros([num_iters, 1])
    for iter in range(num_iters):
        J, grad = costFunction(theta, X, y, lam)
        theta -= alpha * np.reshape(grad, (-1, 1))

        J_history[iter] = J

    return theta, J_history


def predict(theta, X):
    p = X @ theta >= 0
    return p


def getRates(prediction, y):
    tp = np.sum(y * prediction)
    fp = np.sum(np.logical_not(y) * prediction)
    tn = np.sum(np.logical_not(y) * np.logical_not(prediction))
    fn = np.sum(y * np.logical_not(prediction))
    return tp, fp, tn, fn


def getAccuracy(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def getPrecision(tp, fp):
    return tp / (tp + fp)


def getRecall(tp, fn):
    return tp / (tp + fn)


def getFScore(beta, precision, recall):
    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    return f_beta


def binaryPlot(X, y, i=1, j=2):
    marker_size = 15
    plt.scatter(X[:, i], X[:, j], marker_size, c=y)
    plt.scatter
    plt.show()


def boundaryPlot(X, y, theta, i=1, j=2):
    x1 = np.linspace(0, 1, 100)
    x2 = -theta[i] * x1 / theta[j] - theta[0] / theta[j]
    plt.plot(x1, x2, "r")
    binaryPlot(X, y)


# ---------------------------------------------------------


class testModelStats:
    def __init__(self, prediction, y):
        self.prediction = prediction
        self.y = y
        self.tp, self.fp, self.tn, self.fn = getRates(prediction, y)
        self.precision = getPrecision(self.tp, self.fp)
        self.recall = getRecall(self.tp, self.fn)
        self.accuracy = getAccuracy(self.tp, self.fp, self.tn, self.fn)
        self.f1 = getFScore(1, self.precision, self.recall)
        self.f2 = getFScore(2, self.precision, self.recall)

    def __str__(self):
        result = (
            "Accuracy:  "
            + str(self.accuracy)
            + "\nPrecision: "
            + str(self.precision)
            + "\nRecall:    "
            + str(self.recall)
            + "\nF1:        "
            + str(self.f1)
            + "\nF2:        "
            + str(self.f2)
        )
        return result


# ---------------------------------------------------------


class logRegModel:
    # take training dataframe and convert it into inputs and classifications
    # also initialize parameters
    def __init__(self, train, theta=None):
        self.train = train
        self.X = self.train.to_numpy(dtype=np.float32)[:, :-1]
        self.y = self.train.to_numpy(dtype=np.float32)[:, -1]
        self.m, self.n = self.X.shape
        self.X = np.append(np.ones([self.m, 1]), self.X, 1)
        self.y = np.reshape(self.y, (-1, 1))
        self.theta = theta
        if self.theta is None:
            self.theta = np.zeros([1, self.n + 1])
            self.theta = np.reshape(self.theta, (-1, 1))

    # train model parameters using training data
    def trainModel(self, alpha=0.1, num_iters=1000, lam=0.1):
        self.theta, self.J_history = gradientDescent(
            self.X, self.y, self.theta, alpha, num_iters, lam
        )

    # plot 2d projection of points and boundary on the x_i, x_j plane
    def plot2dProjection(self, i=1, j=2):
        marker_size = 15
        plt.scatter(self.X[:, i], self.X[:, j], marker_size, c=self.y)
        plt.scatter
        x1 = np.linspace(0, 1, 100)
        x2 = -self.theta[i] * x1 / self.theta[j] - self.theta[0] / self.theta[j]
        plt.plot(x1, x2, "r")
        plt.show()

    # apply learned parameters to new data
    def predict(self, input):
        X = input.to_numpy(dtype=np.float32)[:, :]
        y = predict(self.theta, X)
        return y

    def test(self, input):
        testModel = logRegModel(input, self.theta)
        prediction = predict(self.theta, testModel.X)
        return testModel, testModelStats(prediction, testModel.y)

