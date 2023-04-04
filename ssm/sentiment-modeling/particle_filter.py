
import json
import numpy as np
import filterpy
import matplotlib.pyplot as plt

from numpy.random import randn, dirichlet

def normalize(v):
    pass

def visualize_emissions_matrix(matrix, numUnique, ItoV, ItoE, emotions):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.xticks(range(numUnique), [ItoV[str(i)] for i in range(numUnique)], rotation=45)
    plt.yticks(range(len(emotions)), [ItoE[str(i)] for i in range(len(emotions))])
    plt.show()

def get_RN_mixture_from_emotions(emotions, p_t):
    # Take a linear combination of the rows of the emotion matrix, weighted by the presence of each emotion in p_t
    emotions_matrix = np.asmatrix(emotions)
    mixture = p_t * emotions_matrix
    return mixture.tolist()[0]

# Taken from: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def create_dirichlet_particles(alpha):
    dirichlet

def main():
    emissions_matrix = np.load("emission_matrices.npy", allow_pickle=True)
    emissions_matrix_header = json.load(open("emission_matrices.json"))
    visualize_emissions_matrix(emissions_matrix, emissions_matrix_header[0], emissions_matrix_header[1], emissions_matrix_header[2], emissions_matrix_header[3])
    print(get_RN_mixture_from_emotions(emissions_matrix, [0.5, 0.3, 0.1, 0.1, 0, 0, 0]))

if __name__ == "__main__":
    main()