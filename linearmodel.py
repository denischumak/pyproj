from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt


class SGDLinearRegressor(RegressorMixin):
    def __init__(
        self,
        lr=1e-4,
        regularization=1.0,
        delta_converged=1e-2,
        max_steps=1000,
        batch_size=64,
    ):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None

    def fit(self, X, Y):
        self.W = np.random.rand(X.shape[1], 1)
        self.b = np.random.rand(1, 1)
        N, D = X.shape
        for steps in range(self.max_steps):
            permutation = np.random.permutation(N)
            X = X[permutation]
            Y = Y[permutation]
            for batch_beg in range(0, N, self.batch_size):
                X_batch = X[batch_beg : np.min((batch_beg + self.batch_size, N))]
                Y_batch = Y[batch_beg : np.min((batch_beg + self.batch_size, N))]
                diff = X_batch @ self.W + self.b - Y_batch
                coeff = 2 / X_batch.shape[0]
                lr_grad_W = self.lr * (
                    coeff * X_batch.T @ diff + 2 * self.regularization * self.W
                )
                lr_grad_b = (self.lr * coeff * np.sum(diff)).reshape(1, 1)
                self.W -= lr_grad_W
                self.b -= lr_grad_b
                if (
                    lng.norm(np.concatenate((lr_grad_W, lr_grad_b)))
                    < self.delta_converged
                ):
                    self.last_conv_steps = steps
                    return self
        self.last_conv_steps = self.max_steps
        return self

    def predict(self, X):
        return X @ self.W + self.b


class AnalyticLinearRegressor(RegressorMixin):
    def __init__(self, regularization=1.0):
        self.regularization = regularization
        self.W = None

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        temp1 = (1 / X.shape[0]) * X.T @ Y
        self.W = (
            np.linalg.inv(
                (1 / X.shape[0]) * X.T @ X + self.regularization * np.eye(X.shape[1])
            )
            @ temp1
        )
        return self
        # X = np.hstack((np.ones((X.shape[0], 1)), X))
        # temp1 = X.T @ Y
        # self.W = (
        #    np.linalg.inv(X.T @ X + self.regularization * np.eye(X.shape[1])) @ temp1
        # )
        # return self

    def predict(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X)) @ self.W




class LinearClassifierSVM(ClassifierMixin):
    def __init__(
        self,
        lr=1e-3,
        regularization=1.0,
        delta_converged=1e-2,
        max_steps=1000,
    ):
        self.lr = lr
        self.regularization = regularization
        self.delta_converged = delta_converged
        self.max_steps = max_steps

        self.W = None

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        N, D = X.shape
        flag = True
        self.W = np.random.rand(D, 1)
        for steps in range(self.max_steps):
            grad_W = 2 * self.regularization * self.W
            grad_W[0] = 0
            for i in range(N):
                margin = Y[i] * self.W.T @ X[i].reshape(D, 1)
                if margin < 1:
                    grad_W += (-Y[i] * X[i]).reshape(D, 1)
            self.W -= self.lr * grad_W
            if self.lr * np.linalg.norm(grad_W) < self.delta_converged:
                self.last_conv_steps = steps
                return self
        self.last_conv_steps = self.max_steps
        return self

    def predict(self, X):
        return np.sign(np.hstack((np.ones((X.shape[0], 1)), X)) @ self.W)
