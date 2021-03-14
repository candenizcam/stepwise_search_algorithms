import itertools
import timeit
from .functions import k_fold
import numpy as np


class BestSubset:
    """This class has two purposes, it contains the static methods that performs various subset selection algorithms
    and it holds results for the traditional methods to ease access later, also it records times for them"""

    def __init__(self, cd, model, exhaustive=False, forward=False, backwards=False, print_loop=False):
        """
        :param cd: ClassificationDataLoader
        :param model: model to be used
        :param exhaustive: if true, exhaustive search is performed
        :param forward: if true forward stepwise search is performed
        :param backwards: if true backwards stepwise search is performed
        :param print_loop: if true, loop performance is printed
        """
        null_model = [(1 - sum(cd.classes) / len(cd.classes), "NULL")]  # calculates the null model as a pair
        self.exhaustive_models = null_model  # initializes models as a pair of accuracy and predictors
        self.forward_models = null_model
        self.backwards_models = null_model
        t1 = timeit.default_timer()  # times are recorded
        if exhaustive:  # exhaustive search
            self.exhaustive_models = self.exhaustive_models + self.exhaustive_search(model, cd)
        t2 = timeit.default_timer()
        if forward:  # fw stepwise
            self.forward_models = self.forward_models + self.forward_stepwise(model, cd, print_loop=print_loop)
        t3 = timeit.default_timer()  # bw stepwise
        if backwards:  # bw stepwise
            self.backwards_models = self.backwards_models + self.backwards_stepwise(model, cd, print_loop=print_loop)
        t4 = timeit.default_timer()
        self.times = {"ex": t2 - t1, "fw": t3 - t2, "bw": t4 - t3}  # a times string is stored for later use

    @staticmethod
    def centre_out_stepwise(model, cd, heads, print_loop=False):
        a0 = [(1 - sum(cd.classes) / len(cd.classes), "NULL")]
        a1 = BestSubset.backwards_stepwise(model, cd, print_loop=print_loop, heads=heads)
        a2 = BestSubset.forward_stepwise(model, cd, print_loop=print_loop, heads=heads)
        return a0 + a1 + a2

    @staticmethod
    def exhaustive_search(model, cd, upto=10000000, print_loop=False, heads=-1):
        """
        This static method performs exhaustive search on a model, and data and returns a list of selections for each
        predictor count
        :param model: the model to be used as ModelInterface
        :param cd: ClassificationDataLoader
        :param upto: an upper bound for the number of predictors to be searched
        :param print_loop: if true, performance is printed
        :param heads: if not -1 selected heads are used
        :return: a list of pairs of best accuracy and keys of predictors
        """
        if heads == -1:
            heads = cd.get_predictor_heads()
        a2 = []
        it_count = min(len(heads), upto)
        for x in range(it_count):
            this_heads = list(itertools.combinations(heads, x + 1))  # for all combinations of relevant predictor no
            a1 = []  # a list of values is taken
            for j in this_heads:
                score = k_fold(cd, model, 10, titles=list(j))
                a1.append((score, j))
            a1.sort(key=lambda x: x[0])  # and the best value is found
            a2.append(a1[-1])  # then recorded
            if print_loop:
                print("x {}% done".format(x / it_count * 100))
        return a2

    @staticmethod
    def forward_stepwise(model, cd, upto=1000000, print_loop=False, heads=-1):
        """
            This static method performs forward stepwise search on a model, and data and returns a list of selections
            for each predictor count
            :param model: the model to be used as ModelInterface
            :param cd: ClassificationDataLoader
            :param upto: an upper bound for the number of predictors to be searched
            :param print_loop: if true, performance is printed
            :param heads: if not -1 selected heads are used
            :return: a list of pairs of best accuracy and keys of predictors
        """
        hl = []
        a2 = []
        if heads != -1:  # search is initialized for different heads entries
            hl = heads
            heads = [i for i in cd.get_predictor_heads() if i not in hl]
        else:
            heads = cd.get_predictor_heads()
        it_count = min(len(heads), upto)
        for x in range(it_count):  # for all predictors
            s1 = []
            m1 = []
            for j in heads:  # each singular addition to the model is searched
                m = hl + [j]
                score = k_fold(cd, model, 10, titles=m)
                s1.append(score)
                m1.append(m)
            best_arg = int(np.argmax(s1))  # best model is found
            a2.append((max(s1), m1[best_arg]))  # best is appended
            heads.pop(best_arg)  # the best predictor is removed from the options
            hl = m1[best_arg]  # and the model is carried
            if print_loop:
                print("fw {}% done".format(x / it_count * 100))
        return a2

    @staticmethod
    def backwards_stepwise(model, cd, upto=1000000, print_loop=False, heads=-1):
        """
            This static method performs backwards stepwise search on a model, and data and returns a list of selections
            for each predictor count
            :param model: the model to be used as ModelInterface
            :param cd: ClassificationDataLoader
            :param upto: an upper bound for the number of predictors to be searched
            :param print_loop: if true, performance is printed
            :param heads: if not -1 selected heads are used
            :return: a list of pairs of best accuracy and keys of predictors
        """
        it_count = min(len(cd.get_predictor_heads()), upto)
        if heads == -1:  # initialized
            heads = cd.get_predictor_heads()
        score = k_fold(cd, model, 10, titles=heads)  # and a score for the full model is found
        a2 = [(score, heads)]
        for x in range(min(len(heads) - 1, upto)):
            combinations = itertools.combinations(a2[0][1], len(a2[0][1]) - 1) # n-1 length combinations are obtained
            a1 = []
            for j in combinations:  # best model is searched
                score = k_fold(cd, model, 10, titles=list(j))
                a1.append((score, j))
            a1.sort(key=lambda xx: xx[0])  # found
            a2 = [a1[-1]] + a2  # and recorded
            if print_loop:
                print("bw {}% done".format(x / it_count * 100))
        return a2
