import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """ Returns the loss as a float."""
    p = np.asarray(p)
    y = np.asarray(y)

    p = p.flatten()
    y = y.flatten()

    num = 2*p@y.T + eps
    den = np.sum(p) + np.sum(y) + eps

    loss = 1 - num/den

    return float(loss)
    
    
