
import numpy as np

def gini_impurity():
    pass

class DecisionTreeNode():

    def __init__(self, labels, data, decision):
        '''
        :params:
        labels: list of "headers" factored in the decision function
        data: table of data (one col per label)
        decision: function to decide left and right childs; returns 2 tuples, for left, right
        '''

        self.labels = labels
        self.data = data
        self.decision = decision

        if decision != None:
            self.left_tuple, self.right_tuple = decision(self.labels, self.data)

            self.left = DecisionTreeNode(self.left_tuple[0], self.left_tuple[1], None)
            self.right = DecisionTreeNode(self.right_tuple[0], self.right_tuple[1], None)

        self.gini = 0

    def grow_child_nodes(self, decision):
        self.decision = decision
        self.left_tuple, self.right_tuple = decision(self.labels, self.data)
        self.left = DecisionTreeNode(self.left_tuple[0], self.left_tuple[1], None)
        self.right = DecisionTreeNode(self.right_tuple[0], self.right_tuple[1], None)

    def get_classification(self, row):
        '''
        :params:
        row: row of data
        '''
        if self.decision == None:
            return self.labels
        return self.decision(self.labels, self.row)


class DecisionTree():

    def __init__(self, labels, data):
        '''
        :params:
        labels: list of length n, with table headers (interpreted as some)
        '''
        self.labels = labels
        self.data = data
    
    def grow_tree(self):
        '''
        :params:
        data: data is presented as a numpy 
        '''
        pass

class ClassificationDT(DecisionTree):

    def __init__(self, labels, data):
        super().__init__(labels, data)
    
    def grow_tree(self):
        return super().grow_tree()