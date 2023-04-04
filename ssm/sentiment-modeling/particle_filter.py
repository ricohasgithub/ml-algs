
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn, dirichlet
from particles import state_space_models as ssm
from particles import distributions as dists

from util import *

# Note in source library particles, x is latent state, y is observations
class SentimentSSM(ssm.StateSpaceModel):

    default_params = {'emotions': [], 'emotions_header': []}

    # Transition model p(a_t | a_t-1): dist of alpha_t at time t, given alpha_{t-1}
    def PX(self, t, xp):
        # Use vector autoregressor on alpha_t
        pass

    # Emission model p(RN_t | a_t): dist of RN_t at time t, given alpha_t and alpha_{t-1}
    def PY(self, t, xp, x):
        # p(RN_t | a_t) = p(RN_t | p_t) * p(p_t | a_t)
        p_t = dirichlet(x)
        rn_dist_given_p = get_RN_mixture_from_emotions(self.emotions, p_t)
        return rn_dist_given_p[int(self.emotions_header[4][t])]

    # Initial distribution
    def PX0(self):
        pass

def main():
    emissions_matrix, emissions_matrix_header = load_emissions_matrices()
    romans_map = dict([(value, key) for key, value in emissions_matrix_header[1].items()])
    emissions_matrix_header.append(romans_map)
    particle_filter = SentimentSSM(emissions_matrix, emissions_matrix_header)
    
if __name__ == "__main__":
    main()