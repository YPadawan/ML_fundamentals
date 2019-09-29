import numpy as np
from sklearn.tree import DecisionTreeClassifier

#TODO still need to fix the rejection sampling
def rejection_sampling(X, y, distribution):
    n, d = X.shape
    X_sample = np.empty_like()
    M = np.max(distribution)
    i = 0
    while i <= 1000:
        U = np.random.random_uniform(1, n, 1)
        V = np.random.uniform(0, M)


def adaboost(X, y, T):
    n, d = X.shape
    prob = 1 / n
    sample_weights = np.empty(n)
    sample_weights[:] = prob
    for t in range(T):
        weak_learner = DecisionTreeClassifier(X, y).fit()
        pred = weak_learner.predict()
        incorrect = np.not_equal(pred, y)
        eps = np.mean(np.average(incorrect, weights=sample_weights))
        a_t = 1/2 ** np.log((1-eps) / eps)
        updated_weights = sample_weights * np.exp(-a_t*y*pred)
        Z_t = updated_weights.sum()
        sample_weights = updated_weights / Z_t
    return np.sign(sample_weights)


