
import numpy as np

def gini_impurity():
    pass

# Boolean (True or False) data
def binary_decision():
    pass

# Real valued data
def continuous_decision():
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

    def get_gini(self):
        return self.gini

class DecisionTree():

    def __init__(self, X, Y):
        '''
        :params:
        X: list of length n, with table headers (interpreted as some set of decisions/questions)
                with the alst column being the final decision/classification/quantification
        Y: table of values for the corresponding header X
        '''
        self.X = X
        self.Y = Y

        self.labels = X[:-1]
        self.label_data = Y[:-1]
        self.result = Y[-1]
    
    def grow_tree(self, node):
        '''
        :params:
        data: data is presented as a numpy 
        '''
        min_gini = 99999999
        min_decision = None

        for i in range(len(self.labels)):

            label = self.labels[i]
            queries = self.label_data[i]

            decision = None
            if type(queries[0]) == "bool":
                decision = binary_decision()
            else:
                decision = continuous_decision()

            gini = gini_impurity()
            if gini < min_gini:
                min_gini = gini
                min_decision = decision

        node.grow_child_nodes(min_decision)
        self.grow_tree(node.left)
        self.grow_tree(node.right)

class ClassificationDT(DecisionTree):

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self.root = self.find_root
        self.grow_tree(self.root)
    
    def find_root(self):
        min_gini = 99999999
        root = None

        for i in range(len(self.labels)):

            label = self.labels[i]
            queries = self.label_data[i]

            decision = None
            if type(queries[0]) == "bool":
                decision = binary_decision()
            else:
                decision = continuous_decision()

            gini = gini_impurity()
            if gini < min_gini:
                min_gini = gini
                root = DecisionTreeNode(self.X, self.Y, decision)

        return root

    def grow_tree(self, node):
        return super().grow_tree(node)