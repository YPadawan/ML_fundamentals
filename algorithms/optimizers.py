import numpy as np


def line_search(X, y, cost_func, p, g, old_loss, w_old):
    """Line search algorithm

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Matrix containing data
    y : array, shape (n_samples,)
        True labels
    w : array, shape (n_features + 1,)
        Weight vector
    cost_func : callable
        Cost function
    p : array, shape (n_features + 1,)
        descent direction
    g : array, shape (n_features + 1,)
        gradient vector
    old_loss : float
        old loss value
    new_loss : float
        new loss value
    w_old : array, shape (n_features,)
        old weight vector

    Returns
    -------
    w : array, shape (n_features + 1,)
        Updated weight vector
    """
    import pdb
    pdb.set_trace()

    alpha = 1e-4
    min_eta = 1e-7

    n, d = X.shape

    # Calcul de la pente au point actuel (float)
    pente = p @ g

    # Définition de la valeur minimale tolérée de eta
    _max = 0.
    for j in range(d+1):
        if np.abs(p[j]) > _max * np.maximum(np.abs(w_old[j]), 1.):
            _max = np.abs(p[j]) / np.maximum(np.abs(w_old[j]), 1.)
    eta_min = min_eta / _max

    # Initialisation de eta à 1
    eta = 1.

    # MAJ de w
    w = w_old + eta * p

    new_loss = cost_func(X, y, w)

    # Boucler tant que la condition d'Armijo n'est pas complète (Eq. 2.18)
    while new_loss > (old_loss + alpha * eta * pente):
        if eta < eta_min:
            w = w_old
            break
        else:
            if eta == 1.:
                # Minimiseur du polynôme d'interpolation de degré 2 (Eq. 2.32)
                eta_tmp = -pente / (2 * (new_loss - old_loss - pente))
            else:
                coeff1 = new_loss - old_loss - eta * pente
                coeff2 = new_loss2 - old_loss - eta2 * pente

                d = np.array([[1 / (eta ** 2), -1 / (eta2 ** 2)], [-eta2 / (eta ** 2), eta / (eta2 ** 2)]])
                c = np.array([coeff1, coeff2]).reshape(2, 1)

                # Calcul des coefficients du polynôme d'interpolation de degré 3 (Eq. 2.33)
                a, b = 1 / (eta - eta2) * np.dot(d, c).flatten()

                if a != 0.:
                    delta = (b ** 2) - 3. * a * pente
                    if delta >= 0.:
                        # minimiseur du polynôme d'interpolation de degré 3 (Eq 2.34)
                        eta_tmp = (-b + np.sqrt(delta)) / (3. * a)
                    else:
                        raise ValueError("rchln:problème d'interpolation")

                else:
                    eta_tmp = -pente / (2.*b)
                if eta_tmp > 0.5*eta:
                    eta_tmp = 0.5*eta
        eta2 = eta
        new_loss2 = new_loss
        eta = np.maximum(eta_tmp, 0.1 * eta)
        w = w_old + eta * p
        new_loss = cost_func(X, y, w)
        print(w)
    return w, new_loss


def conjugate_gradient(X, y, w, cost_func, grd_func, epsilon):
    """Computes the conjugate gradient

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Matrix containing data
    y : array, shape (n_samples,)
        True labels
    w : array, shape (n_features,)
        Weight vector
    cost_func : callable
        Cost function
    grd_func : callable
        Gradient function
    epsilon : float


    Returns
    -------
    Nothing, updates the weight vector with the loss minimizing value

    """
    import pdb
    pdb.set_trace()

    epoque = 0
    n, d = X.shape
    w_old = np.random.random(d+1)
    # p = np.empty_like(w)

    new_loss = cost_func(X, y, w)
    old_loss = new_loss + 2 * epsilon
    g = grd_func(X, y, w)
    # Initialisation de p au gradient en temps 0 (Eq. 2.46)
    p = -g
    print(new_loss)

    while np.abs(old_loss - new_loss) > np.abs(old_loss) * epsilon:
        old_loss = new_loss
        w, new_loss = line_search(X=X, y=y, w=w, cost_func=cost_func,
                                  p=p, g=g, old_loss=old_loss,
                                  new_loss=new_loss, w_old=w_old)
        print(new_loss)
        #         print(w)
        # Calcul du nouveau vecteur gradient (Eq. 2.42)
        h = grd_func(X, y, w)
        dgg = g @ g
        ngg = h @ h

        # Faut-il rajouter la formule de Ribière-Polak (Eq. 2.52)

        # Formule de Fletcher-Reeves (Eq. 2.53)
        beta = ngg / dgg

        w_old = w
        g = h
        # Mise à jour de la direction de descente
        p = -g + beta * p

        print(f"Epoque {epoque} and loss is: {new_loss}")
        print(w)
        epoque += 1
    return w