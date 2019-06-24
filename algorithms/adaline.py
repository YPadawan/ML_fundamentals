import numpy as np

def adaline(X, y, eta=0.1, T=1000):
    n, d = X.shape
    w = np.random.random(d)
    w0 = 0
    t = 0
    h = 0
    while t <= T:
        i = np.random.randint(n)
        yt = y[i]
        Xt = X[i,:]
        h = np.sign((w0 + np.dot(w,Xt.T)))
#         h = activation(h)
        w0 += eta*(yt - h)
        w += eta * (yt - h) * Xt
        t += 1
    return w, w0
    
if __name__ == '__main__':
    # Simulating a table and testing the 
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=25)
    y_actual = np.where(y == 0, -1, y)
    w, w0 = adaline(X, y_actual)
    pred = np.sign((np.dot(w, X.T) + w0))

    # Testing some basic elements of returned values and their type
    assert len(w) == 25
    assert type(w0) == np.dtype('float64')
    assert len(pred) == len(y_actual)
    assert ((pred == 1) | (pred == -1)).all()
