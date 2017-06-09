import logstic as lg
import numpy as np
from mnist import MNIST

mndata = MNIST('./data')
images_train, labels_train = mndata.load_training()
images_train = np.array(images_train) / 256.0
labels_train = np.array(labels_train)
images_test, labels_test = mndata.load_testing()
images_test = np.array(images_test) / 256.0
labels_test = np.array(labels_test)

cls_s = []
def single_cls(a, b):
    train_index = ((labels_train == a) + (labels_train == b))
    train_images = images_train[train_index]
    train_labels = labels_train[train_index]

    lg_cls = lg.logstic(maxiter=500,labels=np.array([a,b]),batch=100)
    lg_cls.fit(train_images, train_labels)
    cls_s.append(lg_cls)
    print('cls ' + str(a) + ' ' + str(b) + ' done')


def predict(X):
    t = np.zeros(10)
    for lg_cls in cls_s:
        t[lg_cls.predict(X)] += 1
    return np.argmax(t)

for i in range(9):
    for j in range(i+1, 10):
        single_cls(i, j)

l = np.shape(images_test)[0]
acc = 0
for i in range(l):
    t = predict(labels_test[i])
    if t == test_labels[i]:
        acc += 1

print(acc * 1.0 / l)
