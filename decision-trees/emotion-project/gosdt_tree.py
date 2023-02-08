import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT

EMOTIONS = {"NONE": 0, "ANGER": 1, "SAD": 2, "FEAR": 3, "IRONY": 4, "LOVE": 5, "JOY": 6}
ROMANS = ["I", "ii", "iii", "IV", "V", "VI", "vii"]

class Chord():

    def __init__(self, roman):
        self.roman = roman
        if roman.isupper():
            # Major emotional distribution; greater weight on 5, 6
            self.emotion_dist = []
        else:
            # Minor emotional distribution; greater weight on 1, 2, 3, 4
            self.emotion_dist = []

class Progression():

    def __init__(self, length, principal_emotion):
        pass

    def generate_sequence(self):
        pass

def generate_fake_musical_data():
    # Generate a distribution of 7 elements for 7 emotions which sum up to 1

    pass