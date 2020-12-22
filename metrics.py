import numpy as np


def accuracy_score(true, pred):
    return np.mean(np.mean((pred == true), axis=1))


def f1_score(true, pred):
    tp = (pred * true).sum()
    precision = tp / (pred.sum())
    recall = tp / (true.sum())

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1.mean().item(), precision.mean().item(), recall.mean().item()
