import autodiff as ad
import numpy as np
import random

class logstic(object):

    def __init__(self, maxiter=100000, learning_rate=0.005 ,labels=np.array([0, 1])):
        self.maxiter = maxiter
        self.labels = labels
        self.learning_rate = learning_rate

    def fit(X, Y):
        x = ad.Variable(name = 'x')
        w = ad.Variable(name = 'w')
        y = ad.Variable(name = 'y')

        p = 1 / (1 + ad.exp_op(-1 * ad.matmul_op(w, x)))

        # cross entropy
        loss = -1 * y * ad.log_op(p) + (1 - y) * ad.log_op(1 - p)

        grad_w, = ad.gradients(loss, [w])

        # SGD
        l = np.shape(X)[0]
        self.coef_ = np.zeros(np.shape(X)[1])
        for i in range(maxiter):
            t = random.choice(range(l))
            loss_val, grad_w_val = executor.run(feed_dict = {x : X[t], w : self.coef_, y : Y[t]})
            self.coef_ = self.coef_ - self.learning_rate * grad_w_val

    def predict(self, X):
        p = np.dot(self.coef_, X)
        if p > 0:
            return self.labels[0]
        else:
            return self.labels[1]
