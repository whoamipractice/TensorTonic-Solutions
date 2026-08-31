import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    #Returns the mean multiclass cross-entropy loss as a Python float.
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    n = yt.shape[0]

    tcp = yp[np.arange(n), yt]

    loss = - np.mean(np.log(tcp))

    return loss
    
