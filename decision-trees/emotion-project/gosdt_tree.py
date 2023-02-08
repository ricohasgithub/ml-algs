import pandas as pd
import numpy as np
import time
import pathlib

from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT

EMOTIONS = ["NONE", "ANGER", "SAD", "FEAR", "IRONY", "LOVE", "JOY"]
EMOTIONS_MAP = {"NONE": 0, "ANGER": 1, "SAD": 2, "FEAR": 3, "IRONY": 4, "LOVE": 5, "JOY": 6}
ROMANS = ["I", "ii", "iii", "IV", "V", "VI", "vii"]
MAJOR_WEIGHTS = [0.04, 0.06, 0.01, 0.02, 0.12, 0.27, 0.48]
MINOR_WEIGHTS = [0.01, 0.31, 0.33, 0.25, 0.05, 0.03, 0.02]

def get_distribution(X, total):
    # Given some list x with discrete samples, return percentage distribution
    return [x / total for x in X]

class Chord():

    def __init__(self, roman):
        self.roman = roman
        if roman.isupper():
            # Major emotional distribution; greater weight on LOVE, JOY
            self.discrete_dist = np.random.choice(EMOTIONS, 100, MAJOR_WEIGHTS)
            self.emotion_dist = get_distribution(self.discrete_dist)
            print(self.emotion_dist)
        else:
            # Minor emotional distribution; greater weight on ANGER, SAD, FEAR, IRONY
            self.discrete_dist = np.random.choice(EMOTIONS, 100, MINOR_WEIGHTS)
            self.emotion_dist = get_distribution(self.discrete_dist)
            print(self.emotion_dist)

class Progression():

    def __init__(self, length, principal_emotion):
        pass

    def generate_sequence(self):
        pass

def generate_fake_musical_data():
    # Generate a distribution of 7 elements for 7 emotions which sum up to 1

    pass