import logstic as lg
import numpy as np
from mnist import MNIST

mndata = MNIST('./data')
images_train, labels_train = mndata.load_training()
images_train = np.array(images_train)
labels_train = np.array(labels_train)
images_test, labels_test = mndata.load_testing()
images_test = np.array(images_test)
labels_test = np.array(labels_test)

index_0 = (labels_train == 0)
train_0 = images_train[index_0]
labels_0 = labels_train[index_0]

index_1 = (labels_train == 1)
train_1 = images_train[index_1]
labels_1 = labels_train[index_1]

X = np.append(train_0, train_1, axis=0)
Y = np.append(labels_0, labels_1, axis=0)

lg_cls = lg.logstic()
lg_cls.fit(X, Y)
