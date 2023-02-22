
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from music_utils import generate_musical_data_as_pd
from sklearn import tree

# Plot confusion matrix, as returned by a scikit classifier; code taken from: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def plot_confusion_matrix(C):
    df_cm = pd.DataFrame(C, range(len(C)), range(len(C[0])))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

def sklearn_tree(N):

    # Get training data
    X, Y = generate_musical_data_as_pd(N)

    # Init tree and fit
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X, Y)

sklearn_tree(10000)