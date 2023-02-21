
import pandas as pd
import numpy as np

from music_utils import generate_musical_data_as_pd
from sklearn import tree

def sklearn_tree(N):

    # Get training data
    X, Y = generate_musical_data_as_pd(N)

    # Init tree and fit
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X, Y)

sklearn_tree(5)