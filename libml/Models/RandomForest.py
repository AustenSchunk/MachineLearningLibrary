import numpy as np
from DecisionTree import ClassificationTree, RegressionTree
from abc import ABCMeta
from scipy.stats import mode
"""
References
----------
.. [1] Murphy, K. P. (2012). Machine learning: A probabilistic perspective.
            Cambridge, MA: MIT Press.
.. [2] The Elements of Statistical Learning: Data Mining, Inference, and Prediction (Second Edition)
            by Trevor Hastie, Robert Tibshirani and Jerome Friedman (2009)
.. [3] https://perso.math.univ-toulouse.fr/motimo/files/2013/07/random-forest.pdf

"""


class RandomForest(metaclass=ABCMeta):
    """
    Attributes
    ----------
    num_trees : the number of trees to be made in the forest
    max_depth : the maximum depth that each tree is allowed to grow
    cost_func : function that determines the cost of each split in the trees
    min_size : the minimum number of data observations needed in each split
    sample_percentage : size of data to be sampled per tree

    Note
    ----
    This class is not to be instantiated. It is simply a base class for the
    classification and regression forest classes
    """

    def __init__(self, num_trees, seed, max_depth, cost_func, min_size, sample_percentage):
        """
        Initializes the random forest

        Parameters
        ----------
        num_trees : the number of trees to be made in the forest
        seed : the seed from which the random sample choices will be made
        max_depth : the maximum depth that each tree is allowed to grow
        cost_func : function that determines the cost of each split in the trees
        min_size : the minimum number of data observations needed in each split
        sample_percentage : size of data to be sampled per tree
        """

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.cost_func = cost_func
        self.min_size = min_size
        self.sample_percentage = sample_percentage
        np.random.seed(seed)

    def fit(self, X, y):
        """
        Grows a forest of decision trees based off the num_trees
        attribute

        Parameters
        ----------
        X : N x D matrix of real or ordinal values
        y : size N vector consisting of either real values or labels for corresponding
        index in X
        """

        data = np.column_stack((X, y))
        self.forest = np.empty(shape=self.num_trees, dtype='object')
        sample_size = int(X.shape[0] * self.sample_percentage)

        for i in range(self.num_trees):
            sample = data[np.random.choice(data.shape[0], sample_size, replace=True)]

            sampled_X = data[:, :data.shape[1] - 1]
            sampled_y = data[:, data.shape[1] - 1]

            if isinstance(self, RegressionForest):
                tree = RegressionTree(
                    max_depth=self.max_depth,
                    min_size=self.min_size,
                    in_forest=True)
            else:
                tree = ClassificationTree(
                    cost_func=self.cost_func,
                    max_depth=self.max_depth,
                    min_size=self.min_size,
                    in_forest=True)

            tree.fit(sampled_X, sampled_y)
            self.forest[i] = tree

    def predict(self, X):
        """
        Predicts the output (y) of a given matrix X

        Parameters
        ----------
        X : numerical or ordinal matrix of values corresponding to some output

        Returns
        -------
        The predict values corresponding to the inputs
        """

        votes = np.zeros(shape=(self.num_trees, X.shape[0]))
        for i, tree in enumerate(self.forest):
            votes[i] = tree.predict(X)

        predictions = np.zeros(shape=X.shape[0])
        if isinstance(self, RegressionForest):
            predictions = votes.mean(axis=0)
        else:
            # print(votes)
            predictions = np.squeeze(mode(votes, axis=0)[0])

        return predictions


class RegressionForest(RandomForest):
    """
    Attributes
    ----------
    num_trees : the number of trees to be made in the forest
    max_depth : the maximum depth that each tree is allowed to grow
    cost_func : function that determines the cost of each split in the trees
    min_size : the minimum number of data observations needed in each split
    sample_percentage : size of data to be sampled per tree
    """

    def __init__(self, num_trees=10, seed=0, max_depth=None, min_size=1, sample_percentage=1):
        """
        Initializes Regression Forest

        Parameters
        ----------
        num_trees : the number of trees to be made in the forest
        seed : the seed from which the random sample choices will be made
        max_depth : the maximum depth that each tree is allowed to grow
        cost_func : function that determines the cost of each split in the trees
        min_size : the minimum number of data observations needed in each split
        sample_percentage : size of data to be sampled per tree
        """

        self.num_trees = num_trees
        self.cost_func = 'mse'
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_percentage = sample_percentage
        super().__init__(
            num_trees=num_trees,
            seed=seed,
            max_depth=max_depth,
            cost_func=self.cost_func,
            min_size=min_size,
            sample_percentage=sample_percentage
            )


class ClassificationForest(RandomForest):
    """
    Attributes
    ----------
    num_trees : the number of trees to be made in the forest
    max_depth : the maximum depth that each tree is allowed to grow
    cost_func : function that determines the cost of each split in the trees
    min_size : the minimum number of data observations needed in each split
    sample_percentage : size of data to be sampled per tree
    """

    def __init__(self, num_trees=10, seed=0, max_depth=None, cost_func='mcr', min_size=1, sample_percentage=1):
        """
        Initializes Regression Forest

        Parameters
        ----------
        num_trees : the number of trees to be made in the forest
        seed : the seed from which the random sample choices will be made
        max_depth : the maximum depth that each tree is allowed to grow
        cost_func : function that determines the cost of each split in the trees
        min_size : the minimum number of data observations needed in each split
        sample_percentage : size of data to be sampled per tree
        """

        self.num_trees = num_trees
        self.cost_func = cost_func
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_percentage = sample_percentage
        super().__init__(
            num_trees=num_trees,
            seed=seed,
            max_depth=max_depth,
            cost_func=cost_func,
            min_size=min_size,
            sample_percentage=sample_percentage
            )
