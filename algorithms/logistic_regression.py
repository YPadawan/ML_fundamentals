import numpy as np


# Le x correspond à formule linéaire de x
# def logistic(x):
#     """Calcul la fonction logistique
#     Parameters
#     ----------
#     x : float,
#         Valeurs auxquelles on applique une transformation logistique
#
#     Returns
#     -------
#     x_tr: float,
#         transformation logistique de x
#     """
#     return 1.0 / (1.0 + np.exp(-x))

def logistic(x):
    """Applique la régression logistique en fonction du signe de x afin de le rendre plus stable

    Parameters
    ----------
    x: float,
        Valeur auxquelles on applique la transformation de x

    Returns
    -------
    x_tr: float,
        Transformation logistique de x

    """

    # if x <= 0:
    #     return np.exp(x) / (1.0 + np.exp(x))
    # else:
    #     return 1.0 / 1 + np.exp(-x)
    return np.where(
        x <= 0,
        np.exp(x) / (1.0 + np.exp(x)),
        1.0 / 1 + np.exp(-x)
    )

def logistic_surrogate_loss(w, X, y):
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
    ps += np.dot(X, w[:-1]).sum()
    # S += stable_logistic(y*ps).sum() / n
    S += logistic(y*ps).sum() / n
    return S


def gradient_logistic_surrogate_loss(w, X, y):
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
    g = np.zeros(d + 1)
    ps = np.dot(X, w[:-1]).sum() + w[-1]
    g[-1] = ((logistic(y * ps) - 1.0) * y).sum()
    g[:d] = np.dot((logistic(y * ps) - 1.0) * y, X).sum()
    g /= n
    return g