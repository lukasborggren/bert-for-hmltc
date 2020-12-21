import numpy as np
from torch import Tensor
from scipy.special import expit
from sklearn.metrics import confusion_matrix


def accuracy(self, out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(
    y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True
):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    # return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    return np.mean(
        ((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1
    ).sum()


def fbeta(
    y_pred: Tensor,
    y_true: Tensor,
    thresh: float = 0.5,
    beta: float = 1,
    eps: float = 1e-9,
    sigmoid: bool = True,
):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2

    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()

    tp = (y_pred * y_true).sum(dim=1)
    prec = tp / (y_pred.sum(dim=1) + eps)
    rec = tp / (y_true.sum(dim=1) + eps)

    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)

    return res.mean().item(), prec.mean().item(), rec.mean().item()


def pairwise_confusion_matrix(y_pred, y_true):
    """Creates a confusion matrix that is used to make pairwise evaluations
    between classes in multi-label classification."""
    # Transform logits to OHE labels
    y_pred = expit(y_pred)
    y_pred = np.where(y_pred > 0.5, 1.0, 0.0)

    # Select rows where only the target category is used
    mask = ((y_true[:, 0] == 1.0) & (y_true[:, 1] == 0.0)) | (
        (y_true[:, 0] == 0.0) & (y_true[:, 1] == 1.0)
    )
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    # mask = (y_pred[:, 0] != 0.0) & (y_pred[:, 1] != 0.0)
    # y_pred = y_pred[mask]
    # y_true = y_true[mask]

    # Represent the labels as binary variable
    y_true = np.where(y_true[:, 0] == 1.0, True, False)
    y_pred = np.where(((y_pred[:, 0] == 1.0) & (y_pred[:, 1] == 0.0)), True, False)

    """y_true, y_pred
    [0,1]=>False, [0,1]=>False => TRUE NEGATIVE
    [0,1]=>False, [1,0]=>True => FALSE POSITIVE
    [0,1]=>False, [1,1]=>False => TRUE NEGATIVE
    [0,1]=>False, [0,0]=>False => TRUE NEGATIVE
    [1,0]=>True,  [0,1]=>False => FALSE NEGATIVE
    [1,0]=>True,  [1,0]=>True => TRUE POSITIVE
    [1,0]=>True,  [1,1]=>False => FALSE NEGATIVE
    [1,0]=>True,  [0,0]=>False => FALSE NEGATIVE"""

    # print(y_true)
    # print(y_pred)

    matrix = confusion_matrix(y_true, y_pred, labels=[True, False])

    return matrix
