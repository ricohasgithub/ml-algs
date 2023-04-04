
import json
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.random import randn, dirichlet
from particles import state_space_models as ssm

def normalize(v):
    return v / norm(v, ord=1)

def visualize_emissions_matrix(matrix, numUnique, ItoV, ItoE, emotions):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.xticks(range(numUnique), [ItoV[str(i)] for i in range(numUnique)], rotation=45)
    plt.yticks(range(len(emotions)), [ItoE[str(i)] for i in range(len(emotions))])
    plt.show()

# Gets p(RN_t | p_t), where p_t is a mixture of emotions
def get_RN_mixture_from_emotions(emotions, p_t):
    # Take a linear combination of the rows of the emotion matrix, weighted by the presence of each emotion in p_t
    emotions_matrix = np.asmatrix(emotions)
    mixture = p_t * emotions_matrix
    # Normalize
    mixture = normalize(mixture.tolist()[0])
    return mixture

def main():
    emissions_matrix = np.load("emission_matrices.npy", allow_pickle=True)
    emissions_matrix_header = json.load(open("emission_matrices.json"))
    visualize_emissions_matrix(emissions_matrix, emissions_matrix_header[0], emissions_matrix_header[1], emissions_matrix_header[2], emissions_matrix_header[3])

if __name__ == "__main__":
    main()