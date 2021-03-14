# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import timeit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

from project import *


def plot_df_map(results):
    """Used to generate matrix plots with DataFrames of boolean"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(results.values.astype(np.bool) == False, cmap="gray")
    ax.set_yticklabels([0] + list(results.index))
    ax.set_xticklabels([0] + list(results))


def data_shrinker(cd, func):
    """
    Shrinks the data for various previously set predictors using a reduction funciton
    :param cd: ClassificationData
    :param func: function for reduction
    :return:
    """
    drop_1 = ["Attr28", "Attr53", "Attr54", "Attr64"]
    drop_2 = ["Attr1", "Attr2", "Attr3", "Attr6", "Attr7", "Attr9", "Attr10", "Attr11", "Attr14", "Attr18", "Attr22",
              "Attr24", "Attr25", "Attr35", "Attr36", "Attr38", "Attr48", "Attr51"]
    drop_3 = ["Attr8", "Attr16", "Attr17", "Attr26", "Attr50"]
    drop_4 = ["Attr4", "Attr33", "Attr40", "Attr46", "Attr63"]
    drop_5 = ["Attr13", "Attr19", "Attr20", "Attr30", "Attr31", "Attr39", "Attr42", "Attr49", "Attr56", "Attr62"]
    drops = [drop_1, drop_2, drop_3, drop_4, drop_5]  # same denominator lists
    dt2 = []
    dt0 = cd.predictors.copy()
    for x, i in enumerate(drops):
        dt0 = dt0.drop(i, axis=1)
        dt = cd.predictors[i]
        dt2.append(func(dt))

    dt1 = pd.concat([dt0] + dt2, axis=1)
    dt1["class"] = cd.classes
    return dt1


def plot_standard_operation(model_name, bs, o2):
    """
    The plotting for the standard operation
    :param model_name: the name for the title of the plot
    :param bs: these are the outputs of the standard opreation, this is best subset
    :param o2: this is the o2 dict
    """
    plt.grid(True)
    plt.xlabel("predictor count")
    plt.ylabel("cross validated accuracy")
    plt.title(model_name)
    plt.plot(list(map(lambda x: x[0], bs.forward_models)), label="fw")
    plt.plot(list(map(lambda x: x[0], bs.backwards_models)), label="bw")
    # plt.plot(list(map(lambda x: x[0], bs.exhaustive_models)), label="x")
    plt.plot(list(map(lambda x: x[0], o2["bs_weight"])), label="weight")
    plt.plot(list(map(lambda x: x[0], o2["bs_freq"])), label="freq")
    plt.plot(list(map(lambda x: x[0], o2["bs_rows"])), label="rows")
    plt.legend()
    plt.savefig("results/lrk_{}.png".format(model_name))
    plt.close()


def standard_operation(model_name, this_model, tree_model, plot=False):
    """
    The standard operation where we first do fw and bw stepwise search, then do a tree, find three heuristic methods and
     apply them, finally return the outputs of these operations, also plot if asked
    :param model_name: the name of the model used
    :param this_model: and the model as ModelInterface
    :param tree_model: the tree model to be used as TreeModelInterface
    :param plot: if true plots
    :return: bs for fw and bw results an a dict for the rest
    """
    bs = BestSubset(cd, this_model, exhaustive=False, forward=True, backwards=True, print_loop=True)
    rs = cd.get_random_split(0.9)  # a random tree split is made, seed is not used to increase variance for results
    tree_time = timeit.default_timer()
    acc = tree_model.fit_and_predict(rs)  # tree is fit
    tree_time2 = timeit.default_timer() # tree time is found
    print("tree time: {}, tree acc: {}:".format(tree_time2 - tree_time, acc)) # and printed, but is often very small
    o2 = { # result dict is supplied with key selections
        "top_weight": tree_model.top_n_weight(len(cd.get_predictor_heads()) // 2 + 2, cd.get_predictor_heads()), #tt1
        "top_freq": tree_model.top_n_freq(len(cd.get_predictor_heads()) // 2 + 2, cd.get_predictor_heads()), #tt2
        "top_rows": tree_model.get_top_rows(cd.get_predictor_heads(), 4) #tt3
    }
    # various heuristic searches are performed
    t1 = timeit.default_timer()
    o2["bs_weight"] = BestSubset.centre_out_stepwise(this_model, cd, list(o2["top_weight"]), print_loop=True) #aa1
    t2 = timeit.default_timer()
    o2["bs_freq"] = BestSubset.centre_out_stepwise(this_model, cd, list(o2["top_freq"]), print_loop=True)
    t3 = timeit.default_timer()
    o2["bs_rows"] = BestSubset.centre_out_stepwise(this_model, cd, list(o2["top_rows"]), print_loop=True)
    t4 = timeit.default_timer()
    o2["times"] = {"ex": t2 - t1, "fw": t3 - t2, "bw": t4 - t3}
    if plot:
        plot_standard_operation(model_name, bs, o2) # optional plotting
    return bs, o2


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    # some other datas for comparison
    # cd = ClassificationData('SomervilleHappinessSurvey2015.csv', "D", encoding="utf-16")
    # cd = ClassificationData('datas/default of credit card clients.csv', "Y", False ,"ID")
    # cd = ClassificationData('datas/aff.csv', "Classes", False)
    # cd = ClassificationData('datas/csv_result-1year.csv', "class",3,"id")
    cd = ClassificationData('datas/newdata.csv', "class", drop_pred=0, set_index="id")
    models = {
        "Lin Disc": ModelInterface(LinearDiscriminantAnalysis()),
        "Quad Disc": ModelInterface(QuadraticDiscriminantAnalysis()),
        "KNN": ModelInterface(KNeighborsClassifier(n_neighbors=5)),
        "Tree": TreeModelInterface(DecisionTreeClassifier())
    }

    # standard operations
    #bsl, o2l = standard_operation("Lin Disc",models["Lin Disc"],models["Tree"],True)
    #bsq, o2q = standard_operation("Quad Disc", models["Quad Disc"],models["Tree"], True)
    #bsk, o2k = standard_operation("KNN", models["KNN"],models["Tree"], True)
    print("hey")

    # data shrinkage
    # dt1 = data_shrinker(cd, lr_killer)
    # dt1.to_csv("datas/newdata.csv")
