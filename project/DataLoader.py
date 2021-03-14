import pandas as pd
import numpy as np
from .PredictionData import PredictionData


class DataLoader:
    """Loads a data and slices predictors if necessary using the path of the data
    """

    def __init__(self, path, drop_pred=0, set_index="", encoding="utf-8"):
        """
        :param path: path to the csv file as a string
        :param drop_pred: the number of top nan predictors to be dropped
        :param set_index: if an index column exists, it is set to be the index column
        :param encoding: if a non utf-8 encoding is used for dataset, it can be entered
        """
        self.data = pd.read_csv(path, encoding=encoding)  # data is read
        self.data = self.data.apply(pd.to_numeric, errors='coerce')  # and coerced to numeric
        if set_index != "":
            self.data = self.data.set_index(set_index)
        self.data = self.nan_preprocess(self.data, drop_pred)

    @staticmethod
    def nan_preprocess(data, count):
        """
        pre processes data to clear nans
        :param data: the data to be processed
        :param count: the amount of predictor columns we want to rid, if this is higher then the nan columns, it only
        drops nan columns
        :return: cleared dataset
        """
        if count <= 0:  # if no dropping request
            return data.dropna()  # nan lines are dropped and df is returned
        t = [sum(data[i].isna()) for i in list(data)]  # nan counts obtained for predictors
        count = min(sum(np.array(t)>0), count)
        worst_indices = [list(data)[i] for i in np.array(t).argsort()[-count:][::-1]]  # top count nan indices obtained
        for i in worst_indices:  # predictors are dropped
            data = data.loc[:, data.columns != i]
        return data.dropna()  # and the nan lines of the rest are dropped

    def get_heads(self):
        """
        A simple function that retuns the keys of the data set
        :return: a list of the keys of the data set
        """
        return list(self.data)

    @staticmethod
    def random_splitter(data, ratio):
        """
        returns the 1-ratio chunk first, rest second
        :param data: data to be split
        :param ratio: ratio of the splitted data
        :return: two datas split by ratio
        """
        ind = np.random.rand(len(data[0])) < ratio
        return [(i[ind], i[~ind]) for i in data]

    @staticmethod
    def index_splitter(data, index):
        """Returns un-indexed first, indexed second, like above but with specified index
        """
        inds = data.index.isin(index)
        return data[~inds], data[inds]


class ClassificationData(DataLoader):
    """Data loader for classification purposes, it inherits the base DataLoader class"""

    def __init__(self, path, class_title, drop_pred=0, set_index="", encoding="utf-8"):
        """
        :param path: as described in the base class
        :param class_title: the title of the classification column
        :param drop_pred: as described in the base class
        :param set_index: as described in the base class
        :param encoding: as described in the base class
        """
        super().__init__(path, drop_pred, set_index, encoding)
        t = list(set(self.get_heads()) - {class_title})
        self.predictors = self.data[t]  # splits the data to predictors
        self.classes = self.data[class_title]  # and class

    def get_predictor_heads(self):
        """
        a simple function that returns the keys of the predictors
        :return: a list of keys
        """
        return list(self.predictors)

    def get_random_split(self, ratio, pred_titles=-1):
        """
        Like random split function of the main class but returns a PredictionData for the split
        :param ratio: as described in the base class
        :param pred_titles: as described in the base class
        :return: PredictionData as the output of the split
        """
        if pred_titles != -1:
            predictors = self.predictors[pred_titles]
        else:
            predictors = self.predictors
        t = self.random_splitter([predictors, self.classes], ratio)
        return PredictionData(t[1][0], t[0][0], t[1][1], t[0][1])

    def get_test_index_split(self, index, pred_titles=-1):
        """Splits by the index values,
        WARNING: THIS DOES NOT SPLIT BY NUMERIC INDEX BUT THE INDEX VALUES OF THE INDEX COLUMN
        :param index: as described in the base class
        :param pred_titles: as described in the base class
        :return: PredictionData as the output of the split
        """
        if pred_titles != -1:
            predictors = self.predictors[pred_titles]
        else:
            predictors = self.predictors
        trp, tep = self.index_splitter(predictors, index)
        trc, tec = self.index_splitter(self.classes, index)
        return PredictionData(trc, trp, tec, tep)
