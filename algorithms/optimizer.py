import numpy as np
import scipy as sp
from scipy.optimize import line_search

#logistic_cost = lambda yhat, ytrue: np.sum(np.log(1 + np.exp(-ytrue * yhat)))
#hinge_cost = lambda yhat, ytrue: np.max(0, 1 - ytrue * yhat)
def logistic_cost(ytrue, yhat):
    return np.log(1 + np.exp(-ytrue*yhat))

def cost_function(yhat, ytrue, method='logistic'):
    
    if method=='logistic':
        cost = np.log(1 + np.exp(-ytrue*yhat))
    elif method=='hinge':
        cost = np.max(0, 1 - ytrue * yhat)))
    elif method=='exponential':
        cost = np.exp(-ytrue*yhat)
    return cost
    

def quasi_newton(X, cost_func, eps):
    d = X.shape[1]
    w = np.random.random(d) # Inititalisation aléatoire des poids

    # Inititalisation de la matrice B0 par un matrice identiité de dimension d
    B0 = np.identity(d, dtype=np.float64)
    
    # Inititalisation de la direction de descente pt
    p0 = -1 * np.gradient(cost_func(w))
    eta = cost_func(w)
    pt = p0

    w_next = w + eta * pt 

    t = 0
    

    while np.linalg.norm(prev_cost, cost):
        eta = cost_func(w_next)
                               
        w_prev = w_next
        w_next= w_prev * eta*pt
        


def stochastic_gradient_descent(X, ytrue, T = 1000, epsilon = 0.1, learning_rate = 0.5): 
	nrow = ytrue.shape[0]
	w = np.zeros(nrow)  # initialisation du vecteur w avec les w_i = 0
	t = 0 

	while (np.linalg.norm(np.gradient(yhat, ytrue)) > epsilon) or (t <= T):
		i = np.random.randint(nrow)
		y_t = ytrue[i,:] 
		X_t = X[i,:]
                prediction = np.dot(X_t, w)
		cost = logistic_cost(yhat_t, y_t)
		w = w - learning_rate * (1/m * X_t.T.dot(prediction - y_t)
		t += 1 
	return w 


def exemple_w():
    w0 = np.zeros(10)
    t = 0
    while (t <= 10):
        print(w0)
        w0 = w0 + 1
        t += 1
    return w0
        
    
