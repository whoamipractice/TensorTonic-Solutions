import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    #Returns the loss as a float.
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    d = np.linalg.norm(a-b, axis = -1)

    l = y*(d**2) + (1-y)*np.maximum(0, margin - d)**2

    if reduction == "sum":
        return float(np.sum(l))
    else:
        return float(np.mean(l))
