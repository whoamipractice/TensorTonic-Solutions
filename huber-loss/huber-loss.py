import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    #Returns the loss as a float.
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    e = yt - yp
    abs_e = np.abs(e)

    loss = np.where(abs_e <= delta, 1/2*abs_e**2, delta*(abs_e - 1/2*delta))

    return float(np.mean(loss))
