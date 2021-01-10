import numpy as np
from sklearn.datasets import make_classification
from algorithms.adaline import adaline

X, y = make_classification(n_samples=100, n_features=25)
y_actual = np.where(y == 0, -1, y)
w, w0 = adaline(X, y_actual)
pred = np.sign((np.dot(w, X.T) + w0))

# Testing some basic elements of returned values and their type
def testing_length():
    assert len(w) == X.shape[1]


def testing_type():
    assert type(w0) == np.dtype('float64')

def output_test():
    assert len(pred) == len(y_actual)
    assert ((pred == 1) | (pred == -1)).all()