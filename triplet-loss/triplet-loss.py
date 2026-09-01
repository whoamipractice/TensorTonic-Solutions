import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    #Returns the loss as a float.
    an = np.asarray(anchor)
    po = np.asarray(positive)
    ne = np.asarray(negative)

    dap = np.linalg.norm(an-po, axis = -1)**2
    dan = np.linalg.norm(an-ne, axis = -1)**2

    loss = np.maximum(0, dap - dan + margin)

    return float(np.mean(loss))
