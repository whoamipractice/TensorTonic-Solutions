import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    #Returns the loss as a float.
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)

    losses = np.maximum(0, margin - yt*ys)

    if reduction == "sum":
        return float(np.sum(losses))
    else:
        return float(np.mean(losses))