import numpy as np


# TODO:  Harmonize var names
def line_search(X, y, w, cost_func, p, g, old_loss, new_loss, w_old):
    """Line search algorithm
    Parameters
    ----------

    X : array_like, shape (n, d),
        Matrice contenant les données
    y : array, shape (n,)
        Vecteur des labels
    w : array, (d+1,)
        Vecteur des poids avec le biais en plus
    cost_func : callable,
        Fonction de coût
    p : array, (d+1,)
        Vecteur de la direction de descente
    g : array (d+1,)
        Vecteur gradient
    """

    alpha = 1e-4
    mineta = 1e-7

    n, d = X.shape
    # initialisation de w_old
    # w_old = w.copy()

    # Initialisation de old_loss et new_loss
    # old_loss = cost_func(X, y, w)
    # new_loss = old_loss

    # Calcul de la pente au point actuel (float)
    pente = p @ g

    # Définition de la valeur minimale tolérée de eta
    _max = 0.
    for j in range(d+1):
        if np.abs(p[j]) > _max * np.maximum(np.abs(w_old[j]), 1.):
            _max = np.abs(p[j]) / np.maximum(np.abs(w_old[j]), 1.)
    etamin = mineta / _max

    # Initialisation de eta à 1
    eta = 1.

    # MAJ de w
    w = w_old + eta * p

    new_loss = cost_func(X, y, w)

    # Boucler tant que la condition d'Armijo n'est pas complète (Eq. 2.18)
    while new_loss > (old_loss + alpha * eta * pente):
        if eta < etamin:
            w = w_old
            break
        else:
            if eta == 1.:
                # Minimiseur du polynôme d'interpolation de degré 2 (Eq. 2.32)
                etatmp = -pente / (2 * (new_loss - old_loss - pente))
            else:
                coeff1 = new_loss - old_loss - eta * pente
                coeff2 = new_loss2 - old_loss - eta2 * pente

                d = np.array([[1 / (eta ** 2), -1 / (eta2 ** 2)], [-eta2 / (eta ** 2), eta / (eta2 ** 2)]])
                c = np.array([coeff1, coeff2]).reshape(2, 1)

                # Calcul des coefficients du polynôme d'interpolation de degré 3 (Eq. 2.33)
                ab = (1 / (eta - eta2) * np.dot(d, c)).reshape(1, -1)
                a, b = ab[0][0], ab[0][1]

                if a != 0.:
                    delta = (b ** 2) - 3. * a * pente
                    if delta >= 0.:
                        # minimiseur du polynôme d'interpolation de degré 3 (Eq 2.34)
                        etatmp = (-b + np.sqrt(delta)) / (3. * a)
                    else:
                        raise ValueError("rchln:problème d'interpolation")

                else:
                    etatmp = -pente / (2.*b)
                    if etatmp > 0.5*eta:
                        etatmp = 0.5*eta
        eta2 = eta
        new_loss2 = new_loss
        eta = np.maximum(etatmp, 0.1 * eta)
        w = w_old + eta * p
        new_loss = cost_func(X, y, w)
    return w, new_loss


def conjugate_gradient(X, y, w, cost_func, grd_func, epsilon):
    epoque = 0
    n, d = X.shape
    w_old = np.random.random(d+1)
    # p = np.empty_like(w)

    new_loss = cost_func(X, y, w)
    old_loss = new_loss + 2 * epsilon
    g = grd_func(X, y, w)
    # Initialisation de p au gradient en temps 0 (Eq. 2.46)
    p = -g
    # print(p)

    while np.abs(old_loss - new_loss) > np.abs(old_loss) * epsilon:
        old_loss = new_loss
        w, new_loss = line_search(X=X, y=y, w=w, cost_func=cost_func,
                                  p=p, g=g, old_loss=old_loss,
                                  new_loss=new_loss, w_old=w_old)
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
        epoque += 1