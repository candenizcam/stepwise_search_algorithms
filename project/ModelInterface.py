from sklearn.tree import plot_tree

from .ConfusionMatrix import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np


class ModelInterface:
    """This interface simplifies using a model to predict data
    model is the model
    pred_func is to classify, particularly useful for linear regression"""

    def __init__(self, model, pred_func=lambda x: x):
        """
        :param model: is the model to be used, usually a Class from sklearn, but anything with the same interface can be
        used
        :param pred_func: the function used to predict the class
        """
        self.model = model
        self.pred_func = pred_func

    def fit_to_model(self, x, y):
        """
        This function can be used to fit data to model with ease, fit_and_predict method is recommended for regular
        operations, it fits the model which can then be used for prediction
        :param x: predictors as a df
        :param y: class as a list
        """
        self.model.fit(x, y)

    def fit_and_predict(self, pred):
        """
        This method fits a PredictionData to the model, then predicts the test predictors, and returns a ConfusionMatrix
        as results
        :param pred: PredictionData class
        :return: ConfusionMatrix about the prediction
        """
        self.model.fit(pred.tr_preds, pred.tr_class)
        p = self.model.predict(pred.te_preds)
        return ConfusionMatrix(pred.te_class.array, self.pred_func(p))


class TreeModelInterface(ModelInterface):
    """This is a model interface for tree models, various methods to extract information from the tree are included"""

    def __init__(self, model):
        """
        :param model:  as base class
        """
        super().__init__(model)

    def plot_the_tree(self, path=""):
        """
        Plots the tree and saves the figure
        :param path: a path to save the figure, if default, plot is not saved
        """
        plt.figure(figsize=(12, 12))
        plot_tree(self.model, fontsize=6)
        if path != "":
            plt.savefig(path, dpi=200)

    def get_predictor_freq(self, predictor_heads):
        """
        Generates a pair that describes the frequency of the keys in the tree
        :param predictor_heads: keys for the predictors
        :return: a pair of values and predictors indexed to the same order
        """
        pbr = self.get_predictors_by_row(predictor_heads)
        return self.predictor_freq(pbr)

    def get_predictor_weight(self, predictor_heads):
        """
        Generates a pair that describes the weights of the keys in the tree
        :param predictor_heads: keys for the predictors
        :return: a pair of values and predictors indexed to the same order
        """
        pbr = self.get_predictors_by_row(predictor_heads)
        return self.predictor_weight(self.model.tree_, pbr)

    def get_nodes_by_row(self):
        """
        Returns a list of lists of node numbers, decorator for a a static function
        :return: list of list of ints
        """
        return self.nodes_by_row(self.model.tree_)

    def get_predictors_by_row(self, heads):
        """
        Gets a list of key lists, ordered by the row they appear, top first, decorator for a static function
        :param heads: a list of the predictor keys to be associated to the nodes
        :return: list of list of keys
        """
        return self.predictors_by_row(self.model.tree_, heads)

    def get_top_rows(self, heads, n):
        """
        A decorator for row picking that includes a top n row selection
        :param heads: keys of the predictors
        :param n: number of rows to be included starting from the top
        :return: a trimmed list of predictors by row
        """
        prd = self.get_predictors_by_row(heads)
        f = [item for sublist in prd[:n] for item in sublist]
        return list(set(f))

    def top_n_weight(self, n, heads):
        """
        Decorator for weight ordered predictors, takes the first n most weighted predictors
        :param n: the number to trim by
        :param heads: keys of the predictor
        :return: trimmed pair of weights and predictors
        """
        v, c = self.get_predictor_weight(heads)
        return v[np.argsort(c)][-n:]

    def top_n_freq(self, n, heads):
        """
        Decorator for freq ordered predictors, takes the first n most weighted predictors
        :param n: the number to trim by
        :param heads: keys of the predictor
        :return: trimmed pair of weights and predictors
        """
        v, c = self.get_predictor_freq(heads)
        return v[np.argsort(c)][-n:]

    @staticmethod
    def nodes_by_row(tree):
        """
        Extracts the nodes for each row and returns as a list of nodes (as list)
        :param tree: the tree to be used
        """
        r = [[0]]
        for x in range(tree.max_depth):
            r2 = []
            for j in r[-1]:
                lv = tree.children_left[j]
                if lv == -1:
                    continue
                rv = tree.children_right[j]
                r2 = r2 + [lv, rv]
            r.append(r2)
        return r

    @staticmethod
    def predictors_by_row(tree, heads):
        """
        The method that extracts nodes by row for static usage
        :param tree: tree to be used
        :param heads: keys
        :return: list of list of predictors by row
        """
        r = TreeModelInterface.nodes_by_row(tree)
        pr = [tree.feature[i] for i in r]
        return [[heads[j] for j in i if j != -2] for i in pr]

    @staticmethod
    def predictor_freq(pbr, row_no=-1):
        """
        Calculates the predictor frequency at a row, or all if -1
        :param pbr: predictors by row information, obtained from the relevant method
        :param row_no: row no from top
        :return: a tuple of fequency and keys
        """
        if row_no == -1:
            f = [item for sublist in pbr for item in sublist]
        else:
            f = pbr[row_no]
        values, counts = np.unique(f, return_counts=True)
        return values, counts

    @staticmethod
    def predictor_weight(tree, pbr, row_no=-1):
        """
            Calculates the predictor weight of the row no, or all if -1
            :param tree: the tree to be used
            :param pbr: predictors by row information, obtained from the relevant method
            :param row_no: row no from top
            :return: a tuple of weights and keys
        """
        if row_no == -1:
            f = [item for sublist in pbr for item in sublist]
        else:
            f = pbr[row_no]
        values, ind = np.unique(f, return_index=True)
        return values, tree.n_node_samples[ind]
