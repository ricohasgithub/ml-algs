
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn, dirichlet
from particles import state_space_models as ssm

from util import *

# Note in source library particles, x is latent state, y is observations
class SentimentSSM(ssm.StateSpaceModel):

    # Transition model: dist of X_t at time t, given X_{t-1}
    def PX(self, t, xp):
        pass

    # Emission model: dist of Y_t at time t, given X_t and X_{t-1}
    def PY(self, t, xp, x):
        pass

    # Initial distribution
    def PX0(self):
        pass

def main():
    emissions_matrix, emissions_matrix_header = load_emissions_matrices()
    
if __name__ == "__main__":
    main()