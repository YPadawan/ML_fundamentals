import numpy as np


def line_search(X, y, w, wold, cost_func, p, g, oldloss, newloss):
    alpha = 1e-7
    mineta = 1e-4

    n, d = X.shape
    # initialisation de wold
    wold = w
    #     oldloss = cost_func(X, y, w)
    #     newloss = oldloss

    # Calcul de la pente au point actuel (float)
    pente = p @ g

    # Définition de la valeur minimale tolérée de eta
    _max = 0.
    #         if np.abs(p[j]) > _max  * np.maximum(np.abs(w[0]), 1):
    #             _max = np.maximum(np.abs(w[j]), 1.0)
    #     etamin = _mineta / _max
    for j in range(d + 1):
        if np.abs(p[j]) > _max * np.maximum(np.abs(wold[j]), 1.):
            print(np.abs(p[j]))
            _max = np.abs(p[j]) / np.maximum(np.abs(wold[j]), 1.)
            print(_max)
    etamin = mineta / _max

    # Initialisation de eta à 1
    eta = 1.
    w = wold + eta * p
    newloss = cost_func(X, y, w)

    # Boucler tant que la condition d'Armijo n'est pas complète (Eq. 2.18)
    while newloss > (oldloss + alpha * eta * pente):
        if eta < etamin:
            w = wold
            break
        else:
            if eta == 1.:
                # Minimiseur du polynôme d'interpolation de degré 2 (Eq. 2.32)
                etatmp = -pente / 2. * (oldloss - (oldloss * pente))
            else:
                coeff1 = newloss - oldloss - eta * pente
                coeff2 = newloss2 - oldloss - eta2 * pente

                # Calcul des coefficients du polynôme d'interpolation de degré 3 (Eq. 2.33)
                a = (coeff1 / (eta ** 2) - coeff2 / (eta ** 2)) / (eta - eta2)
                b = (-eta2 * coeff1 / (eta ** 2) + eta * coeff2 / (eta2 ** 2)) / (eta - eta2)
                if a != 0.:
                    delta = (b ** 2) - 3. * a * pente
                    if delta >= 0.:
                        # minimiseur du polynôme d'interpolation de degré 3 (Eq 2.34)
                        etatmp = (-b + np.sqrt(delta)) / (3.0 * a)
                    else:
                        #                         print("problème d'interpolation")
                        raise ValueError("rchln :problème d'interpolation")

                else:
                    etatmp = -pente / (2. * b)
                    if etatmp > 0.5 * eta:
                        etatmp = 0.5 * eta
        eta2 = eta
        newloss2 = newloss
        eta = np.maximum(etatmp, 0.1 * eta)
        w = wold + eta * p
        newloss = cost_func(X, y, w)
    return newloss


def conjugate_gradient(X, y, w, cost_func, grd_func, epsilon):
    epoque = 0
    n, d = X.shape
    wold = np.empty_like(w)
    p = np.empty_like(w)
    newloss = cost_func(X, y, w)
    oldloss = newloss + 2 * epsilon
    g = grd_func(X, y, w)
    # Initialisation de p au gradient en temps 0 (Eq. 2.46)
    p = -g
    print(p)

    while np.abs(oldloss - newloss) > np.abs(oldloss) * epsilon:
        newloss = line_search(X=X, y=y, w=w, wold=wold, cost_func=cost_func,
                              p=p, g=g, oldloss=oldloss, newloss=newloss)
        print(newloss)
        # Calcul du nouveau vecteur gradient (Eq. 2.42)
        h = grd_func(X, y, w)
        dgg = g @ g
        ngg = h @ h

        # Faut-il rajouter la formule de Ribière-Polak (Eq. 2.52)

        # Formule de Fletcher-Reeves (Eq. 2.53)
        beta = ngg / dgg

        wold = w
        g = h
        # Mise à jour de la direction de descente
        p = -g + beta * p

        print(f"Epoque {epoque} and loss is: {newloss}")
        epoque += 1