
import numpy as np

class HMM():

    def __init__(self,
                 transition_matrix=None, 
                 emissions_matrix=None, 
                 hidden_states=None,
                 observed_states=None):
        
        # Assign instance variables
        self.transition_matrix = transition_matrix if transition_matrix is not None else []
        self.emissions_matrix = emissions_matrix if emissions_matrix is not None else []
        self.hidden_states = hidden_states if hidden_states is not None else []
        self.observed_states = observed_states if observed_states is not None else []


    def get_viterbi_path(self, observed_sequence):

        # Returns the most likely path of hidden states given a sequence of observations

        # List of lists (acting as iterable tuples) to keep track of the max probabilities which ends at each hidden state so far
        probabilities = []
        # Generate initial distributions based on priors
        curr_observation_index = self.observed_states.index(observed_sequence[0])
        probabilities.append([self.priors[self.hidden_states.index(hidden_state)]*self.emissions_matrix[curr_observation_index]
                              for hidden_state in self.hidden_states])
        
        # For each new observation, append highest probabilitiy to each hidden state's corresponding entry in list probabilities
        for i in range(1, len(observed_sequence)):
            # Get the "prior"; posterior at t-1
            last_states_probs = probabilities[-1]
            # Get the index of the current observation's token
            curr_observation_index = self.observed_states.index(observed_sequence[i])

        # Get path back
        most_likely_path = []
        for p in probabilities:
            most_likely_path.append(self.hidden_states[p.index(max(p))])

        return most_likely_path
