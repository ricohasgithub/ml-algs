
import pandas as pd
import numpy as np

ROMANS = ["I", "i", "V"]
ROMAN_MAP = {"I": 0, "i": 1, "V": 2}
ROMAN_TRANSMISSION = [[0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0],
                      [0.5, 0.5, 0.0]]
ROMAN_MAP_TOKENS = {"S": 0, "I": 1, "i": 2, "V": 3}

EMOTIONS = ["NONE", "ANGER", "SAD", "FEAR", "IRONY", "LOVE", "JOY"]
EMOTIONS_MAP = {"NONE": 0, "ANGER": 1, "SAD": 2, "FEAR": 3, "IRONY": 4, "LOVE": 5, "JOY": 6}

# Weights for emotions
MAJOR_WEIGHTS = [0.04, 0.06, 0.01, 0.02, 0.12, 0.27, 0.48]
MINOR_WEIGHTS = [0.01, 0.31, 0.33, 0.25, 0.05, 0.03, 0.02]

def get_distribution(X, total):
    # Given some list x with discrete samples, return percentage distribution
    sum_X = [0, 0, 0, 0, 0, 0, 0]
    for x in X:
        sum_X[EMOTIONS_MAP[x]] += 1
    return [s_x / total for s_x in sum_X]

class Chord():

    def __init__(self, roman):
        self.roman = roman
        if roman.isupper():
            # Major emotional distribution; greater weight on LOVE, JOY
            self.discrete_dist = np.random.choice(EMOTIONS, 100, p=MAJOR_WEIGHTS)
            self.emotion_dist = get_distribution(self.discrete_dist, 100)
        else:
            # Minor emotional distribution; greater weight on ANGER, SAD, FEAR, IRONY
            self.discrete_dist = np.random.choice(EMOTIONS, 100, p=MINOR_WEIGHTS)
            self.emotion_dist = get_distribution(self.discrete_dist, 100)

    def print_chord(self):
        print(self.roman, ": ", self.emotion_dist)

def generate_chord_progression(n):
    # Total sequence of chords has length n

    CHORDS = [Chord(r) for r in ROMANS]
    start = np.random.choice(CHORDS, 1)[0]

    progression = [start.emotion_dist]
    progression_roman = [start.roman]

    for i in range(n-1):

        # Generate new distribution for each chord
        CHORDS = [Chord(r) for r in ROMANS]

        last_roman = progression_roman[i]
        next_chord = np.random.choice(CHORDS, 1, p=ROMAN_TRANSMISSION[ROMAN_MAP[last_roman]])[0]

        progression.append(next_chord.emotion_dist)
        progression_roman.append(next_chord.roman)

    return progression, progression_roman

def generate_musical_data(m, n):
    # Generate a distribution of 7 elements for 7 emotions which sum up to 1
    # Returns a n x m matrix of emotions
    data = []
    data_romans = []
    for i in range(m):
        data.append(generate_chord_progression(n)[0])
        data_romans.append(generate_chord_progression(n)[1])
    return data, data_romans

# Generate musical data for t = N time steps
def generate_musical_data_as_pd(N):

    # Start token for X is S
    x = []
    y = []

    CHORDS = [Chord(r) for r in ROMANS]

    # Pick starting chord
    start = np.random.choice(CHORDS, 1)[0]
    x.append(["S", *start.emotion_dist])
    y.append(start.roman)

    for t in range(1, N):

        # Generate new distribution for each chord
        CHORDS = [Chord(r) for r in ROMANS]

        last_roman = y[t-1]
        next_chord = np.random.choice(CHORDS, 1, p=ROMAN_TRANSMISSION[ROMAN_MAP[last_roman]])[0]

        x_t = [last_roman, *next_chord.emotion_dist]
        x.append(x_t)
        y.append(next_chord.roman)

    # Convert from string romans to their integer encodings
    for t in range(len(x)):
        x[t][0] = ROMAN_MAP_TOKENS[x[t][0]]
    for t in range(len(y)):
        y[t] = ROMAN_MAP_TOKENS[y[t]]

    X = pd.DataFrame(x, columns=["harmony_t-1", *EMOTIONS])
    Y = pd.DataFrame(y, columns=["harmony_t"])

    return X, Y