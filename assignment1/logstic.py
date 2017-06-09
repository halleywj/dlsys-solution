import autodiff as ad
import numpy as np
import random

class logstic(object):

    def __init__(self, maxiter=1000, learning_rate=0.0005 ,labels=np.array([0, 1]), batch=500):
        self.maxiter = maxiter
        self.labels = labels
        self.learning_rate = learning_rate
        self.batch = batch

    def fit(self, X, Y):
        x = ad.Variable(name = 'x')
        w = ad.Variable(name = 'w')
        y = ad.Variable(name = 'y')

        p = 1 / (1 + ad.exp_op(0 - ad.matmul_op(w, x)))

        # cross entropy
        loss = 0 - y * ad.log_op(p) - (1 - y) * ad.log_op(1 - p)

        grad_w, = ad.gradients(loss, [w])

        # SGD
        length = np.shape(X)[0]
        self.num_feature = np.shape(X)[1]
        executor = ad.Executor([loss, grad_w])
        self.coef_ = np.zeros((1, self.num_feature))
        for i in range(self.maxiter):
            grad = np.zeros((1, self.num_feature))
            loss = 0
            for j in range(self.batch):
                t = random.choice(range(length))
                loss_val, grad_w_val = executor.run(feed_dict = {x : X[t].reshape((self.num_feature, 1)), w : self.coef_, y : Y[t]})
                grad = grad + grad_w_val
                loss = loss + loss_val
            self.coef_ = self.coef_ - self.learning_rate * grad
            if i % 100 == 0:
                print(loss)
                # print(grad)

    def predict(self, X):
        p = 1 / (1 + np.exp(-1 * np.dot(self.coef_, X.reshape(self.num_feature, 1))))
        if p < 0.5:
            return self.labels[0]
        else:
            return self.labels[1]
