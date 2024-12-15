import numpy as np


class GeneralOptimizer:
    def __init__(self, delta, max_epochs, desc="generalOptimizer"):
        self.delta_ = delta
        self.max_epochs_ = max_epochs
        self.desc_ = desc
        self.grad_b_ = 0
        self.grad_W_ = 0

    def checkConverged_(self, old_grad_b, old_grad_W):
        return (old_grad_b - self.grad_b_) ** 2 + np.sum(
            (old_grad_W - self.grad_W_) ** 2
        ) < self.delta_ * self.delta_

    def calcLoss_(self, X, y, W, b):
        N = X.shape[0]
        return np.sum((y - X @ W - b) ** 2) / N


class GradientDescent(GeneralOptimizer):
    def __init__(self, lr=1e-2, delta=1e-4, max_epochs=1000):
        super().__init__(delta, max_epochs, "Gradient Descent")
        self.lr_ = lr

    def optimize(self, X, y):
        self.weight_trace_ = []  # TODELETE
        N, D = X.shape
        b = np.random.standard_normal()
        W = np.random.standard_normal((D, 1))
        loss_step_values_ = []
        steps_count_ = 0
        coeff = 2 / N

        for _ in range(self.max_epochs_):
            loss_step_values_.append(self.calcLoss_(X, y, W, b))
            self.weight_trace_.append([b, W[0, 0]])  # TODELETE

            old_grad_b = self.grad_b_
            old_grad_W = self.grad_W_

            temp_mat = X @ W + b - y

            self.grad_b_ = coeff * np.sum(temp_mat)
            self.grad_W_ = coeff * X.T @ temp_mat

            if self.checkConverged_(old_grad_b, old_grad_W):
                steps_count_ = len(loss_step_values_)
                return W, b, loss_step_values_, steps_count_

            b -= self.lr_ * self.grad_b_
            W -= self.lr_ * self.grad_W_

        steps_count_ = len(loss_step_values_)
        return W, b, loss_step_values_, steps_count_


class StochasticGradientDescent(GeneralOptimizer):
    def __init__(self, lr=1e-2, delta=1e-4, batch_size=32, max_epochs=1000):
        super().__init__(delta, max_epochs, "Stochastic Gradient Descent")
        self.batch_size_ = batch_size
        self.lr_ = lr

    def optimize(self, X, y):
        self.weight_trace_ = []  # TODELETE
        N, D = X.shape
        b = np.random.standard_normal()
        W = np.random.standard_normal((D, 1))
        loss_step_values_ = []
        steps_count_ = 0
        coeff = 2 / N

        for _ in range(self.max_epochs_):

            permutation = np.random.permutation(N)
            X = X[permutation]
            y = y[permutation]

            for batch_begins in range(0, N, self.batch_size_):
                loss_step_values_.append(self.calcLoss_(X, y, W, b))
                self.weight_trace_.append([b, W[0, 0]])  # TODELETE

                X_batch = X[batch_begins : np.min((batch_begins + self.batch_size_, N))]
                y_batch = y[batch_begins : np.min((batch_begins + self.batch_size_, N))]

                old_grad_b = self.grad_b_
                old_grad_W = self.grad_W_

                temp_mat = X_batch @ W + b - y_batch
                # lr = 1 / (steps_count_ + 1)
                self.grad_b_ = coeff * np.sum(temp_mat)
                self.grad_W_ = coeff * X_batch.T @ temp_mat

                if self.checkConverged_(old_grad_b, old_grad_W):
                    steps_count_ = len(loss_step_values_)
                    return W, b, loss_step_values_, steps_count_

                b -= self.lr_ * self.grad_b_
                W -= self.lr_ * self.grad_W_

        steps_count_ = len(loss_step_values_)
        return W, b, loss_step_values_, steps_count_


class Momentum(GeneralOptimizer):
    def __init__(self, lr=1e-2, alpha=0.9, delta=1e-4, batch_size=32, max_epochs=1000):
        super().__init__(delta, max_epochs, "Momentum")
        self.batch_size_ = batch_size
        self.lr_ = lr
        self.alpha_ = alpha

    def optimize(self, X, y):
        self.weight_trace_ = []  # TODELETE
        N, D = X.shape
        b = np.random.standard_normal()
        W = np.random.standard_normal((D, 1))
        h_b = 0
        h_W = np.zeros((D, 1))
        loss_step_values_ = []
        steps_count_ = 0
        coeff = 2 / N

        for _ in range(self.max_epochs_):

            permutation = np.random.permutation(N)
            X = X[permutation]
            y = y[permutation]

            for batch_begins in range(0, N, self.batch_size_):
                loss_step_values_.append(self.calcLoss_(X, y, W, b))
                self.weight_trace_.append([b, W[0, 0]])  # TODELETE

                X_batch = X[batch_begins : np.min((batch_begins + self.batch_size_, N))]
                y_batch = y[batch_begins : np.min((batch_begins + self.batch_size_, N))]

                old_grad_b = self.grad_b_
                old_grad_W = self.grad_W_

                temp_mat = X_batch @ W + b - y_batch

                self.grad_b_ = coeff * np.sum(temp_mat)
                self.grad_W_ = coeff * X_batch.T @ temp_mat

                if self.checkConverged_(old_grad_b, old_grad_W):
                    steps_count_ = len(loss_step_values_)
                    return W, b, loss_step_values_, steps_count_

                h_b = self.alpha_ * h_b + self.lr_ * self.grad_b_
                h_W = self.alpha_ * h_W + self.lr_ * self.grad_W_

                b -= self.lr_ * h_b
                W -= self.lr_ * h_W

        steps_count_ = len(loss_step_values_)
        return W, b, loss_step_values_, steps_count_


class Adam(GeneralOptimizer):
    def __init__(
        self,
        lr = 1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        delta=1e-4,
        batch_size=32,
        max_epochs=1000,
    ):
        super().__init__(delta, max_epochs, "Adam")
        self.batch_size_ = batch_size
        self.lr_ = lr
        self.beta1_ = beta1
        self.beta2_ = beta2
        self.eps_ = eps


    def optimize(self, X, y):
        self.weight_trace_ = []  # TODELETE
        N, D = X.shape
        b = np.random.standard_normal()
        W = np.random.standard_normal((D, 1))
        m_b = 0
        m_W = 0
        v_b = 0
        v_W = 0
        loss_step_values_ = []
        steps_count_ = 0
        coeff = 2 / N

        for _ in range(self.max_epochs_):

            permutation = np.random.permutation(N)
            X = X[permutation]
            y = y[permutation]

            for batch_begins in range(0, N, self.batch_size_):
                loss_step_values_.append(self.calcLoss_(X, y, W, b))
                self.weight_trace_.append([b, W[0, 0]])  # TODELETE

                X_batch = X[batch_begins : np.min((batch_begins + self.batch_size_, N))]
                y_batch = y[batch_begins : np.min((batch_begins + self.batch_size_, N))]

                old_grad_b = self.grad_b_
                old_grad_W = self.grad_W_

                temp_mat = X_batch @ W + b - y_batch

                self.grad_b_ = coeff * np.sum(temp_mat)
                self.grad_W_ = coeff * X_batch.T @ temp_mat

                if self.checkConverged_(old_grad_b, old_grad_W):
                    steps_count_ = len(loss_step_values_)
                    return W, b, loss_step_values_, steps_count_

                k = len(loss_step_values_)
                m_b = self.beta1_ * m_b + (1 - self.beta1_) * self.grad_b_
                m_b /= (1 - self.beta1_ ** k)
                m_W = self.beta1_ * m_W + (1 - self.beta1_) * self.grad_W_
                #m_W /= (1 - self.beta1_ ** k)
                
                v_b = self.beta2_ * v_b + (1 - self.beta2_) * self.grad_b_ ** 2
                v_b /= (1 - self.beta2_ ** k)
                v_W = self.beta2_ * v_W + (1 - self.beta2_) * self.grad_W_ ** 2
                #v_W /= (1 - self.beta2_ ** k)
                
                b -= self.lr_ * m_b / (np.sqrt(v_b) + self.eps_)
                W -= self.lr_ * m_W / (np.sqrt(v_W) + self.eps_)
                

        steps_count_ = len(loss_step_values_)
        return W, b, loss_step_values_, steps_count_


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.loss_step_values_ = []
        self.steps_count_ = 0

    def fit(self, X, y, optimizer):
        self.coef_, self.intercept_, self.loss_step_values_, self.steps_count_ = (
            optimizer.optimize(X, y)
        )
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
