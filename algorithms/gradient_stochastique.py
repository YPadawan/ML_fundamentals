import numpy as np 





# fonction de coût partielle (calcule l'erreur d'une fonction entre les ytrue(le vrai)
# et le yhat (le y obtenu) ) 



def logistic_cost(ytrue, yhat): 
	return np.log(1 + np.exp(-ytrue*yhat))

print(f"Resultat: {logistic_cost(1,1)} ") 

# nombre d'itérations maximal
#T = 

# précision								 
# epsilon = 	

# pas d'apprentissage 

						 
def Gradient_Stochastique( ytrue, yhat, T = 1000, epsilon = 0.1, eta = 0.5): 
	nrow = ytrue.shape[0]
	w = np.zeros(nrow)							# initialisation du vecteur w avec les w_i = 0
	t = 0 

	while (np.linalg.norm(np.gradient(yhat, ytrue)) > epsilon) or (t <= T):
		i = np.random.randint(nrow)
		y_t = ytrue[i] 
		yhat_t = yhat[i]
		cost = logistic_cost(yhat_t, y_t)

		w = w - (eta * np.gradient(cost))

		t += 1 

	return w 

ytrue = np.random.randint(2,size=10)

yhat = np.random.randint(2,size=10)


print(f"Resultat : {Gradient_Stochastique(ytrue, yhat)} ")



