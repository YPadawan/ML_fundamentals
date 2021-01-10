import numpy as np
from sklearn.datasets import make_classification

# Perceptron function testing ssh with proper account 4
def perceptron(X, y, eta=0.1, T=1000):
    """Fits the perceptron algorithm
    Parameters
    ----------
    X: {array like, sparse matrix}, shape (n_samples, n_features)
       Samples
    y: array, shape (n_samples)
       True labels
    eta: float, learning step
    T: integer, number of iterations

    Returns
    -------
    tuple: 
        w: array, shape (n_features)
        w0: float
    Vector of weigths with a bias w0
      
    """
    n, d = X.shape
    w = np.zeros(d)
    w0 = 0.
    t = 0
    while t <= T:
        i = np.random.randint(n)
        y_t = y[i] 
        X_t = X[i,:]
        if (y_t * ((w @ X_t.T) + w0)) <= 0:
            w0 += eta*y_t
            w += eta * y_t * X_t
        t +=1
    return w, w0

if __name__ == '__main__':
    # Simulating a table and testing the algorithm
    X, y = make_classification(n_samples=100, n_features=25)
    y_actual = np.where(y == 0, -1, y)
    w, w0 = perceptron(X, y_actual)
    pred = np.sign((np.dot(w, X.T) + w0))

    # Testing some basic elements of returned values and their type
    assert len(w) == 25
    assert type(w0) == np.dtype('float64')
    assert len(pred) == len(y_actual)
    assert ((pred == 1) | (pred == -1)).all()
    
