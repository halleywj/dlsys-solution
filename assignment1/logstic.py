import autodiff as ad
import numpy as np
import random

class logstic(object):

    def __init__(self, maxiter=100000, learning_rate=0.005 ,labels=np.array([0, 1])):
        self.maxiter = maxiter
        self.labels = labels
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        x = ad.Variable(name = 'x')
        w = ad.Variable(name = 'w')
        y = ad.Variable(name = 'y')

        p = 1 / (1 + ad.exp_op(-1 * ad.matmul_op(w, x)))

        # cross entropy
        loss = -1 * y * ad.log_op(p) + (1 - y) * ad.log_op(1 - p)

        grad_w, = ad.gradients(loss, [w])

        # SGD
        length = np.shape(X)[0]
        num_feature = np.shape(X)[1]
        executor = ad.Executor([loss, grad_w])
        self.coef_ = np.random.rand(1, num_feature)
        for i in range(self.maxiter):
            t = random.choice(range(length))
            loss_val, grad_w_val = executor.run(feed_dict = {x : X[t].reshape((num_feature, 1)), w : self.coef_, y : Y[t]})
            self.coef_ = self.coef_ - self.learning_rate * grad_w_val
            print(self.coef_)

    def predict(self, X):
        p = np.dot(self.coef_, X)
        if p > 0:
            return self.labels[0]
        else:
            return self.labels[1]
