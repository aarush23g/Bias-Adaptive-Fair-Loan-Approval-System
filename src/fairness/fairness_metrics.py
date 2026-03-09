import numpy as np


def demographic_parity(preds, sensitive):
    """
    Difference in positive prediction rates between groups
    """

    groups = np.unique(sensitive)

    rates = []

    for g in groups:
        mask = sensitive == g
        rates.append(preds[mask].mean())

    return rates[0] - rates[1]


def equal_opportunity(preds, y_true, sensitive):
    """
    TPR difference between groups
    """

    groups = np.unique(sensitive)

    tprs = []

    for g in groups:

        mask = sensitive == g

        y_g = y_true[mask]
        p_g = preds[mask]

        positives = y_g == 1

        if positives.sum() == 0:
            tprs.append(0)
        else:
            tpr = ((p_g == 1) & positives).sum() / positives.sum()
            tprs.append(tpr)

    return tprs[0] - tprs[1]


def equalized_odds(preds, y_true, sensitive):
    """
    TPR + FPR difference
    """

    groups = np.unique(sensitive)

    tprs = []
    fprs = []

    for g in groups:

        mask = sensitive == g

        y_g = y_true[mask]
        p_g = preds[mask]

        positives = y_g == 1
        negatives = y_g == 0

        if positives.sum() == 0:
            tpr = 0
        else:
            tpr = ((p_g == 1) & positives).sum() / positives.sum()

        if negatives.sum() == 0:
            fpr = 0
        else:
            fpr = ((p_g == 1) & negatives).sum() / negatives.sum()

        tprs.append(tpr)
        fprs.append(fpr)

    return abs(tprs[0] - tprs[1]) + abs(fprs[0] - fprs[1])