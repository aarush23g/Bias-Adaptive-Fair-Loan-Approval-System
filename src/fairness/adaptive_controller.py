import numpy as np


class AdaptiveFairnessController:

    def __init__(self, alpha=0.02, target=0.02):

        self.alpha = alpha
        self.target = target
        self.lambda_t = 0.0

    def update(self, fairness_violation):

        # Only update if violation exceeds tolerance
        if abs(fairness_violation) > self.target:

            update = self.alpha * fairness_violation

            self.lambda_t = self.lambda_t + update

        # Clip lambda for stability
        self.lambda_t = np.clip(self.lambda_t, -2, 2)

        return self.lambda_t


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_controller(probs, lambda_t):

    """
    Convert lambda into threshold using sigmoid
    """

    threshold = sigmoid(lambda_t)

    preds = (probs >= threshold).astype(int)

    return preds, threshold