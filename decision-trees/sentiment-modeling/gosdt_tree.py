
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from music_utils import generate_musical_data_as_pd

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from gosdt import GOSDT

'''
Music Sentiment Classification Task:
    - Given a chord and some set of emotions associated with it, what is the most likely chord that comes next?
'''

# Plot confusion matrix, as returned by a scikit classifier; code taken from: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def plot_confusion_matrix(C, title, plot=True):
    df_cm = pd.DataFrame(C, range(len(C)), range(len(C[0])))
    plt.figure(figsize=(5,3))
    ax = plt.axes()
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, ax=ax, annot=True, annot_kws={"size": 16}) # font size
    ax.set_title(title)
    if plot: plt.show()

def fit_tree(N, X, Y, tree_type):

    # Get training data and split 90/10 (train/test)
    if (X is None) and (Y is None):
        X, Y = generate_musical_data_as_pd(N)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    # Init tree and fit
    decision_tree = tree_type()
    decision_tree.fit(X_train, Y_train)

    # Get predictions on training and testing data
    preds_train = decision_tree.predict(X_train)
    preds_test = decision_tree.predict(X_test)

    # Get accuracy scores
    train_acc = decision_tree.score(X_train, Y_train)
    test_acc = decision_tree.score(X_test, Y_test)

    # Visualize tree (if possible)
    if tree_type == tree.DecisionTreeClassifier:
        tree.plot_tree(decision_tree)

    # Confusion matrix visual
    C_train = confusion_matrix(Y_train, preds_train)
    C_test = confusion_matrix(Y_test, preds_test)
    plot_confusion_matrix(C_train, f"Training data w/ accuracy {train_acc}", plot=False)
    plot_confusion_matrix(C_test, f"Testing data w/ accuracy {test_acc}", plot=False)

def sklearn_tree(N, X, Y):

    # Get training data and split 90/10 (train/test)
    if (X is None) and (Y is None):
        X, Y = generate_musical_data_as_pd(N)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    # Init tree and fit
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)

    # Get predictions on training and testing data
    preds_train = decision_tree.predict(X_train)
    preds_test = decision_tree.predict(X_test)

    # Get accuracy scores
    train_acc = decision_tree.score(X_train, Y_train)
    test_acc = decision_tree.score(X_test, Y_test)

    # Visualize trees
    tree.plot_tree(decision_tree)

    # Confusion matrix visual
    C_train = confusion_matrix(Y_train, preds_train)
    C_test = confusion_matrix(Y_test, preds_test)
    plot_confusion_matrix(C_train, f"Training data w/ accuracy {train_acc}", plot=False)
    plot_confusion_matrix(C_test, f"Testing data w/ accuracy {test_acc}", plot=False)

def boosted_tree(N, X, Y):

    # Get training data and split 90/10 (train/test)
    if (X is None) and (Y is None):
        X, Y = generate_musical_data_as_pd(N)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    # Init gradient boosted tree
    boost_tree = GradientBoostingClassifier()
    boost_tree.fit(X_train, Y_train)

    # Get predictions on training and testing data
    preds_train = boost_tree.predict(X_train)
    preds_test = boost_tree.predict(X_test)

    # Get accuracy scores
    train_acc = boost_tree.score(X_train, Y_train)
    test_acc = boost_tree.score(X_test, Y_test)

    # Confusion matrix visual
    C_train = confusion_matrix(Y_train, preds_train)
    C_test = confusion_matrix(Y_test, preds_test)
    plot_confusion_matrix(C_train, f"Training data w/ accuracy {train_acc}", plot=False)
    plot_confusion_matrix(C_test, f"Testing data w/ accuracy {test_acc}", plot=False)

def gosdt_tree(N, X, Y):

    # Get training data and split 90/10 (train/test)
    if (X is None) and (Y is None):
        X, Y = generate_musical_data_as_pd(N)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    config = {
            "regularization": 0.02,
            "depth_budget": 5,
            "time_limit": 30,
            "similar_support": False
        }

    gosdt_tree = GOSDT(config)
    gosdt_tree.fit(X_train, Y_train)

    train_acc = gosdt_tree.score(X_train, Y_train)
    test_acc = gosdt_tree.score(X_test, Y_test)

    preds_train = gosdt_tree.predict(X_train)
    preds_test = gosdt_tree.predict(X_test)

    # Confusion matrix visual
    C_train = confusion_matrix(Y_train, preds_train)
    C_test = confusion_matrix(Y_test, preds_test)
    plot_confusion_matrix(C_train, f"Training data w/ accuracy {train_acc}", plot=False)
    plot_confusion_matrix(C_test, f"Testing data w/ accuracy {test_acc}", plot=False)


fit_tree(100, None, None, tree.DecisionTreeClassifier)
# gosdt_tree(100, None, None)
plt.show()