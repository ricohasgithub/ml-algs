
import numpy as np
import numpy.random as random
import scipy
import filterpy
import matplotlib.pyplot as plt

from util import *

# Note in source library particles, x is latent state, y is observations
class SentimentSSM():

    def __init__(self, emotions, emotions_header):
        self.emotions = emotions
        self.emotions_header = emotions_header

    # Transition model p(a_t | a_t-1): dist of alpha_t at time t, given alpha_{t-1}
    def p_trans(self, t, a_t, a_t1):
        # Use vector autoregressor on alpha_t
        # p(a_t | a_t-1) = MVN(VAR(a_t-1), P)
        # VAR: a_t = phi*(a_t-1)
        pass

    # Emission model p(RN_t | a_t): dist of RN_t at time t, given alpha_t
    def p_emit(self, t, RN_t, a_t):
        # p(RN_t | a_t) = p(RN_t | p_t) * p(p_t | a_t)
        p_t = random.dirichlet(a_t)
        rn_dist_given_p = get_RN_mixture_from_emotions(self.emotions, p_t)
        return rn_dist_given_p[int(self.emotions_header[4][RN_t])]

    def predict(self):
        pass

    def update(self, particles, weights):
        distance = np.linalg.norm(particles[:, 0:2], axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.e-300      # avoid round-off to zero
        weights /= sum(weights) # normalize
    
    # Simulate state and observation processes
    def simulate(self, T):
        a = []
        for t in range(T):
            prior_a = self.p_init() if t == 0 else self.p_trans(t, a[-1])
            a.append()

def main():
    emissions_matrix, emissions_matrix_header = load_emissions_matrices()
    romans_map = dict([(value, key) for key, value in emissions_matrix_header[1].items()])
    emissions_matrix_header.append(romans_map)
    particle_filter = SentimentSSM(emissions_matrix, emissions_matrix_header)
    
if __name__ == "__main__":
    main()