
import numpy as np

class HMM():

    def __init__(self,
                 transition_matrix=None, 
                 emission_matrix=None, 
                 hidden_states=None,
                 observed_states=None):
        
        # Assign instance variables
        # 2d matrix where (r, c) = P(r|c)
        self.transition_matrix = transition_matrix if transition_matrix is not None else []
        self.emission_matrix = emission_matrix if emission_matrix is not None else []
        self.hidden_states = hidden_states if hidden_states is not None else []
        self.observed_states = observed_states if observed_states is not None else []


    def get_viterbi_path(self, observed_sequence):

        # Returns the most likely path of hidden states given a sequence of observations

        # List of lists (acting as iterable tuples) to keep track of the max probabilities which ends at each hidden state so far
        probabilities = []
        # Generate initial distributions based on priors
        curr_observation_index = self.observed_states.index(observed_sequence[0])
        probabilities.append([self.priors[self.hidden_states.index(hidden_state)]*self.emission_matrix[self.hidden_states.index(hidden_state)][curr_observation_index]
                               for hidden_state in self.hidden_states])
        
        # For each new observation, append highest probabilitiy to each hidden state's corresponding entry in list probabilities
        for i in range(1, len(observed_sequence)):
            # Get the "prior"; posterior at t-1
            last_states_probs = probabilities[-1]
            # Get the index of the current observation's token
            curr_observation_index = self.observed_states.index(observed_sequence[i])
            next_state_probs = []
            for hidden_state in self.hidden_states:
                hidden_state_index = self.hidden_states.index(hidden_state)
                max_hidden_state = max([last_state_prob*self.transition_matrix[self.hidden_states[self.hidden_states.index(i_hidden_state)]][hidden_state_index]*self.emission_matrix[hidden_state_index][curr_observation_index]
                                        for i_hidden_state in self.hidden_states for last_state_prob in last_states_probs])
                next_state_probs.append(max_hidden_state)
            probabilities.append(next_state_probs)

        # Get path back
        most_likely_path = []
        for p in probabilities:
            most_likely_path.append(self.hidden_states[p.index(max(p))])

        return most_likely_path
