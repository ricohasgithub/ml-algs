
import numpy as np

class HMM():

    def __init__(self,
                 transmissions_matrix=None, 
                 emissions_matrix=None, 
                 hidden_states=None,
                 observed_states=None):
        
        # Assign instance variables
        self.transmissions_matrix = transmissions_matrix if transmissions_matrix is not None else []
        self.emissions_matrix = emissions_matrix if emissions_matrix is not None else []
        self.hidden_states = hidden_states if hidden_states is not None else []
        self.observed_states = observed_states if observed_states is not None else []


    def get_most_likely_path(self, observed_sequence):


        pass