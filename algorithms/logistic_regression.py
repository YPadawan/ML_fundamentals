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


#TODO: Add documentation
def logistic_surrogate_loss(w, X, y):
    # Computing the dot product
    n, d = X.shape
    ps = np.dot(X, w[:-1]) + w[-1]
    yps = y * ps
#     loss = np.where(yps > 0,
#                    np.log(1 + np.exp(-yps)),
#                    (-yps + np.log(1 + np.exp(yps))))
#     loss = logistic(yps)
    loss = np.log(1. + np.exp(-yps))
#     loss = loss.sum()
#     loss /= n
    return np.mean(loss)


def gradient_log_surrogate_loss(w, X, y):
    # defining dim variables
    n, d = X.shape
    z = X.dot(w[:-1]) + w[-1]
    z = logistic(y * z)
    z0 = (z - 1) * y

    # initiating g: gradient vector
    g = np.zeros(d + 1)
    # Computing dot product
    g[:-1] = X.T.dot(z0)
    g[-1] = z0.sum()
    g /= n
    return g