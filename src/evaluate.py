import numpy as np
from sklearn.metrics import confusion_matrix

def find_best_threshold(probs, y):
    best_t = 0.5
    best_cost = float("inf")

    for t in np.linspace(0.01, 0.99, 50):
        preds = (probs > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

        cost = fn * 5000 + fp * 50   # business logic

        if cost < best_cost:
            best_cost = cost
            best_t = t

    return best_t, best_cost