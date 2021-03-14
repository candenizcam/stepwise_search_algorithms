import numpy as np


class ConfusionMatrix:
    """Generates the confusion matrix based on a classification data where
        r0 = right 0 prediction, wrong 0 prediction
        r1 = wrong 1 prediction, right 1 prediction
    """

    def __init__(self, real, pred):
        """
        :param real: the actual values
        :param pred: the prediction values
        """
        real, pred = np.array(real), np.array(pred)
        self.tp = np.sum((real == 0) * (pred == 0))
        self.tn = np.sum((real == 1) * (pred == 1))
        self.fp = np.sum((real == 0) * (pred == 1))
        self.fn = np.sum((real == 1) * (pred == 0))

    def matrix(self):
        """
        this method returns the matrix
        :return: the confusion matrix as an array
        """
        return np.array([[self.tp, self.fp], [self.fn, self.tn]])

    def tpr(self):
        """
        returns the true positive rate
        :return: a float
        """
        return self.tp/(self.tp + self.fn)

    def tnr(self):
        """
        returns the true negative rate
        :return: a float
        """
        return self.tn/(self.tn + self.fp)

    def acc(self):
        """
        returns the accuracy
        :return: a float
        """
        return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)

    def null_acc(self):
        """
        returns the null accuracy
        :return: a float
        """
        return max(self.tp+self.fn, self.tn + self.fp)/(self.tp + self.fn + self.tn + self.fp)
