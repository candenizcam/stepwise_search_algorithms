import itertools

import numpy as np
from sklearn.linear_model import LinearRegression


def k_fold(cd, model, n, titles=-1, seed=3):
    """
    Performs k fold cross validation by classification accuracy and returns the mean of the results
    :param cd: ClassificationDataLoader
    :param model: ModelInterface to be used
    :param n: split number
    :param titles: keys to selectively use the method, -1 means all
    :param seed: random seed, necessary for cross method comparison
    :return: a float that is the mean of accuracy for folds
    """
    indices = np.copy(cd.data.index)  # indices are obtained from the data
    np.random.seed(seed)  # randomization is stabilised
    np.random.shuffle(indices)  # a random shuffle is made
    splits = np.array_split(indices, n)  # data is split to n equal numbered splits
    q = []
    for x, i in enumerate(splits):  # k fold is done
        pred = cd.get_test_index_split(i, titles)  # data is separated by index
        cm = model.fit_and_predict(pred)  # and prediction is done
        q.append(cm.acc())  # results are added to the list
    return np.mean(q)  # mean is returned


def lr_killer(dt):
    """
    reduces the number of predictors using linear regression R^2 analysis
    :param dt: the data set to perform operations
    :return: the smaller dataset
    """
    lr = LinearRegression()
    iter_number = len(list(dt))  # a good estimation for iteration number
    for x in range(iter_number):
        pair = -1  # to ease breaking outer loop
        for i in itertools.permutations(list(dt), 2):
            ind1, ind2 = i[0], i[1]
            lr.fit(dt[[ind1]], dt[ind2])  # a fit is made
            s = lr.score(dt[[ind1]], dt[ind2])  # R^2 is calculated
            if s > 0.95:  # if it is very high
                pair = i  # a target is set
                break
        if pair != -1:  # if a target is found
            dt = dt.drop(pair[1], axis=1)  # the well predictable predictor is removed
        else:  # if not, breaks the loop
            break
        print("{}% complete".format(x / iter_number))  # to show progress
    return dt


def pcr(dt, pca, lr, tag="N"):
    """
    WARNING THIS METHOD DOES NOT WORK, BUT IS NOT ERASED FOR COMPLETENESS
    performs principle component regression with linear regression to find optimal targets
    :param dt: the data set to perform operations
    :param pca: pca to be used
    :param lr: linear regression model to be used
    :param tag: the tag to be used as a prefix for the new data columns
    :return: the pcr applied dataset
    """
    iter_number = len(list(dt))  # a good estimation for iteration number
    for x in range(iter_number):
        m = -1  # to ease breaking outer loop
        for i in itertools.combinations(list(dt), 2):
            v1, v2 = i[0], i[1]  # a pair of predictors
            lr.fit(dt[[v1]], dt[v2])  # first lr, applied to all data
            p1 = lr.predict(dt[[v1]])
            rss1 = np.sum((p1 - dt[v2]) ** 2)  # first rss
            lr.fit(dt[[v2]], dt[v1])  # second lr, applied to all data
            p2 = lr.predict(dt[[v2]])
            rss2 = np.sum((p2 - dt[v1]) ** 2)  # second rss
            v = max(rss2, rss1) / min(rss1, rss2)  # ratio of rss is obtained as a ratio >1
            if v < 2:  # if the ratio is less than 2
                m = [v, i]  # a target is recorded
                break  # and the loop is broken
        if m != -1:  # if a pcr target is found
            pair = m[1]
            dt["{}{}".format(tag, x)] = pca.fit_transform(dt[list(pair)])  # pcr is made and implemented to the new row
            dt = dt.drop(list(pair), axis=1)  # and the source of pcr is removed from the data frame
        else:  # if not, breaks the loop
            break
        print("{}% complete".format(x / iter_number))  # to show progress
    return dt
