import scipy as sp
import numpy as np

# Defining logistic return function
def logistic(x):
    return (1.0 / (1.0 + np.exp(-x)))

# Estimating gradient surrogate loss (Eq 3.17 dans le bouquin)
def GradientLogisticSurrogateLoss(w, X, y):
    """Computing gradient surrogate loss
    Parameters:
     w (vec): weight vector
     train (array): Training set
    """
    n_samples, n_features = X.shape
    g = np.empty_like(w)
    
    for j in range(ncols+1):
        g[j] = 0.0
    for i in range(1, nrows+1):
        ps = w[0]
        for j in range(1, ncols+1):
            ps += w[j] * X[i][j]
        g[0] += logistic(y[i]*ps)-1.0) * y[i]
            for j in range(1, ncols+1):
                g[j] += (logistic(y[i]*ps)-1.0)*y[i]*X[i][j]
    for j in range(ncols+1):
        g[j]/= nrows
    return g

# Calcul de la fonction de coût logistique (Eq 3.16 du livre)

def LogisticSurrogateLoss(w, X, y):
    S = 0
    ps = 0

    nrows, ncols = X.shape
    for i in range(1, nrows+1):
        ps = w[0]
        for j in range(1, ncols+1):
            ps+=w[j]*X[i][j]
        S += log(1.0 + np.exp(-y[i]*ps))
    S /= nrows
    return S

