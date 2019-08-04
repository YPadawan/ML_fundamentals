import numpy as np


# Le x correspond à formule linéaire de x
def logistic(x):
    return (1.0 / (1.0 + np.exp(-x)))


def logistic_surrogate_loss(X, y, w):
    """ Calcul de la fonction de coût logistique
    Paramètres
    -----------
    X: matrix, or sparse array shape (n, d)
    y: array, shape (n,)
        True labels
    w: array, shape (d+1,)
        Weight vectors (the +1 is for the intercept)
    Renvoie
    -------
    loss : float,
        Valeur de la fonction de coût
    """
    n, d = X.shape
    S = 0.
    ps = 0.
    ps += np.dot(X,w[:d]).sum()
    S += (logistic(y*ps)).sum() / n
    return S


def gradient_logistic_surrogate_loss(X, y, w):
    """Calcul du vecteur gradient Eq. (3.17) avec le biais en plus
    Paramètres
    -----------
    X: matrix, or sparse array shape (n, d)
    y: array, shape (n,)
        True labels
    w: array, shape (d,)
        Weight vectors
    w0: scalar,
        bais
    Renvoie
    -------
    grad : array, shape (d,)
        Vecteur gradient de la fonction de coût logistique
    """
    n, d = X.shape
    S = 0.
    g = np.zeros(d + 1)
    ps = 0.
    ps += np.dot(X, w[:d]).sum()
    g[-1] = 0.
    for i in range(n):
        g[-1] += ((logistic(y * ps) - 1.0) * y).sum()
        g[:d] += np.dot((logistic(y * ps) - 1.0) * y, X)

    g /= n
    return g
