import numpy as np

from scipy.optimize import line_search


def back_tracking_line_search(x0, f, g, p, alpha0=1, rho=1, c=1e-4):

    alpha = alpha0
    xk = x0
    old_fval = f(xk)
    while f(xk + alpha * p) > f(xk) + c*alpha*np.dot(g(xk).T, p):
        alpha = rho*alpha
    new_fval = f(xk + alpha * p)
    return alpha, old_fval, new_fval


def conjugate_gradient(x0, obj_func, grd_func, args=()):
    f0 = obj_func(x0, *args)
    g0 = grd_func(x0, *args)
    p = -g0
    x = x0
    g = g0
    epoque = 0
    while np.linalg.norm(g) >= 0.000005:
        alpha, fc, gc, new_loss, old_loss, new_slope = line_search(f=obj_func,
                                                                   myfprime=grd_func,
                                                                   xk=x, pk=p, gfk=g, old_fval=f0, args=args)
        x = x + alpha * p
        h = grd_func(x, *args)
        dgg = np.linalg.norm(g)
        ngg = np.linalg.norm(h)

        # Fletcher-Reeves's beta (Eq 2.53)
        #         beta = ngg / dgg

        # Ribière-Polak beta
        delta = np.dot(h, (h - g))
        beta = max(0, delta / dgg)

        g = h
        p = -g + beta * p
        print(f"Epoque {epoque} and loss is: {new_loss}")
        epoque += 1
    return x

