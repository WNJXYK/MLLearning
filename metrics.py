import numpy as np
from sklearn import metrics


def hamming_loss(label, pred):
    n, m = label.shape
    loss = np.sum([np.sum(label[i] ^ pred[i]) for i in range(n)])
    return loss / n / m


def ranking_loss(label, prob):
    n, m = label.shape
    loss = 0
    for i in range(n):
        prob_positive = prob[i, label[i, :] > 0.5]
        prob_negative = prob[i, label[i, :] < 0.5]
        sum = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    sum += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + sum * 1.0 / (label_negative * label_positive)
    return loss / n


def one_error(label, prob):
    n, m = label.shape
    loss = 0
    for i in range(n):
        pos = np.argmax(prob[i, :])
        loss += label[i, pos] < 0.5
    return loss / n
