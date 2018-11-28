import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

print("Loading data...")
train_data = np.loadtxt(fname='data/fashion-mnist_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt(fname='data/fashion-mnist_testsmall.csv', delimiter=',', skiprows=1)
print("Very data, much mnist, wow")

neigh = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')  # euclidean is the default distance

print("Many training...")
neigh.fit(train_data[:, :-1], train_data[:, -1])

print("Very predict...")
prediction = neigh.predict(test_data[:, :-1])

print("Classification summary")
print(classification_report(prediction, test_data[:, -1]))