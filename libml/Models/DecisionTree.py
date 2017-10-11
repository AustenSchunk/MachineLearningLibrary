import numpy as np
import numpy.linalg as la
from abc import ABCMeta
import scipy.stats as stats
"""
References
----------
.. [1] Murphy, K. P. (2012). Machine learning: A probabilistic perspective.
            Cambridge, MA: MIT Press.
.. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
            and Regression Trees", Wadsworth, Belmont, CA, 1984.
"""


class DecisionTree(metaclass=ABCMeta):
    """
    Attributes
    ----------
    cost_func : cost function for tree
    max_depth : maximum depth of tree
    min_size : minimum size of the data being split
    min_impurity : minimum percent of homogenous data required for the building
    of certain branch of tree to be halted // used for classification
    min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
    in_forest : specifies whether tree will be a part of a random forest

    Note
    ----
    This class is not to be instantiated. It is simply a base class for the
    classification and regression tree classes
    """

    def __init__(self, cost_func, max_depth, min_size, min_impurity, min_cost, in_forest):
        """
        Initializes an abstract Decision tree

        Parameters
        ----------
        cost_func : cost function for tree
        max_depth : maximum depth of tree
        min_size : minimum size of the data being split
        min_impurity : minimum percent of homogenous data required for the building
        of certain branch of tree to be halted // used for classification
        min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
        in_forest : specifies whether tree will be a part of a random forest
        """

        if isinstance(self, RegressionTree):
            self.cost_func = self.regression_cost
        else:
            if cost_func == 'mcr':
                self.cost_func = self.misclassification_rate_cost
            elif cost_func == 'entropy':
                self.cost_func = self.entropy
            elif cost_func == 'gini':
                self.cost_func = self.gini
            else:
                raise ValueError("Must select proper cost function")

        self.max_depth = max_depth
        self.min_size = min_size
        self.min_impurity = min_impurity
        self.min_cost = min_cost
        self.in_forest = in_forest

    def regression_cost(self, y):
        """
        Determines the cost of choosing current values based on the mean squared error

        Parameters
        ----------
        y : output value for each data observation

        Returns
        -------
        mean squared error of choosing current prediction
        """

        if (y.size == 0):
            return 0
        y_hat = np.mean(y)
        diff = y - y_hat
        return np.square(diff).sum()

    def misclassification_rate_cost(self, y):
        """
        Determines the cost of choosing current values based on the
        misclassification rate

        Parameters
        ----------
        y : output value for each data observation

        Returns
        -------
        misclassification rate
        """

        prediction = 0
        for label in np.unique(y):
            cur = np.sum(y == label) / y.shape[0]
            if cur > prediction:
                prediction = cur
        return 1 - prediction

    def entropy(self, y):
        """
        Calculates the entropy of choosing a given label for
        current node

        Parameters
        ----------
        y : labels for a corresponding data observation

        Returns
        -------
        entropy of given labels
        """

        dist = np.empty(np.unique(y).shape)
        for label in range(dist.shape[0]):
            dist[label] = np.sum(y == label) / y.shape[0]
        dist += np.finfo('float').resolution
        return -np.sum(dist * np.log(dist))

    def gini(self, y):
        """
        Computes gini index of the distribution of y

        Parameters
        ----------
        y : labels for a corresponding data observation

        Returns
        -------
        gini index of given distribution 
        """

        dist = np.empty(np.unique(y).shape)
        for label in range(dist.shape[0]):
            dist[label] = np.sum(y == label) / y.shape[0]
        return 1 - np.sum(np.square(dist))


    def split(self, X, y, node):
        """
        Using in training to determine the best split for the data and
        return the appropriate 'left' and 'right' splits

        Parameters
        ----------
        X : matrix consisting of data that is in the current node
        y : vector of labels corresponding to each row in X
        node : current node where data is being split

        Returns
        -------
        Two tuples, the first contains the data observations whose threshold feature value
        was less than or equal to the threshold. The second contains data observations whose
        threshold feature value was greater than the threshold.
        """

        split_threshold = None
        split_feature = None
        min_response = None

        possible_samples = range(X.shape[1])
        if (self.in_forest):

            if isinstance(self, RegressionTree):
                sample_features = np.random.choice(possible_samples, size = int(X.shape[1] / 3))
            else:
                sample_features = np.random.choice(possible_samples, size = int(np.sqrt(X.shape[1])))
        else:
            sample_features = possible_samples

        for feature in sample_features:
            for thresh in np.unique(X[:, feature]):
                left = y[X[:, feature] <= thresh]
                right = y[X[:, feature] > thresh]
                response = self.cost_func(left) + self.cost_func(right)
                if (min_response is None or response < min_response):
                    split_threshold = thresh
                    split_feature = feature
                    min_response = response

        node.threshold = split_threshold
        node.feature = split_feature
        lower_mask = X[:, split_feature] <= split_threshold
        upper_mask = X[:, split_feature] > split_threshold
        return (X[lower_mask], y[lower_mask]), (X[upper_mask], y[upper_mask])

    def not_worth_splitting(self, data_left, data_right, depth):
        """
        Determines if the suggested split is worth doing

        Parameters
        ----------
        data_left : tuple containing the data from the left and its label
        data_right : tuple containing the data from the right and its label
        depth : the current depth of the node where the suggested split will occur

        Returns
        -------
        boolean that will indicate whether the split should occur
        """

        if self.max_depth and depth > self.max_depth:
            return True
        if not isinstance(self, ClassificationTree) and self.cost_reduction(data_left, data_right) < self.min_cost:
            return True
        if data_left[0].size < self.min_size or data_right[0].size < self.min_size:
            return True

        return False

    def cost_reduction(self, data_left, data_right):
        """
        Parameters
        ----------
        data_left : tuple containing the data from the left and its label
        data_right : tuple containing the data from the right and its label

        Returns
        -------
        total amount of cost reduction from choosing to split
        """

        y_total = np.hstack((data_left[1], data_right[1]))
        total_norm = la.norm(y_total)
        left_norm = la.norm(data_left[1])
        right_norm = la.norm(data_right[1])

        total_cost = self.cost_func(y_total)
        normalized_left = (left_norm / total_norm) * self.cost_func(data_left[1])
        normalized_right = (right_norm / total_norm) * self.cost_func(data_right[1])

        return total_cost - (normalized_left + normalized_right)

    def test_purity(self, y):
        """
        Tests labels in node to see if they are all the same

        Parameters
        ----------
        y : current labels in the node

        Returns
        -------
        true or false, indicating whether all labels are the same
        """

        common = stats.mode(y)[0][0]
        return np.sum(y == common) == y.size

    def grow_tree(self, node, X, y, depth):
        """
        Recursive method used to grow a decision tree

        Parameters
        ----------
        node : the node that will split the data
        X : features of the data
        y : labels of the data
        depth : current depth of the the node

        Returns
        -------
        current node created in tree
        """

        if isinstance(self, RegressionTree):
            node.mean_dist = np.mean(y)

        else:
            node.mean_dist = common = stats.mode(y)[0][0]
        if y.size < 2:
            return node
        if isinstance(self, ClassificationTree) and self.test_purity(y):
            return node

        data_left, data_right = self.split(X, y, node)

        if self.not_worth_splitting(data_left, data_right, depth):
            return node

        left = DecisionNode()
        right = DecisionNode()
        node.left = self.grow_tree(left, data_left[0], data_left[1], depth + 1)
        node.right = self.grow_tree(right, data_right[0], data_right[1], depth + 1)

        return node

    def single_prediction(self, x, node):
        """
        Predictions the output of a single node

        Parameters
        ----------
        x : single data observation
        node : current node being traversed to find
        node that will contain the prediction

        Returns
        -------
        prediction based on x
        """

        if x[node.feature] is None or (not node.left and not node.right):
            return node.mean_dist

        go_left = x[node.feature] <= node.threshold

        if (go_left and node.left):
            return self.single_prediction(x, node.left)
        if (not go_left and node.right):
            return self.single_prediction(x, node.right)
        return node.mean_dist

    def fit(self, X, y):
        """
        Fits the data to a decision tree

        Parameters
        ----------
        X : N x D matrix of real or ordinal values
        y : size N vector consisting of either real values or labels for corresponding
        index in X
        """

        node = DecisionNode()
        self.root = self.grow_tree(node, X, y, 0)

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

        predictions = np.zeros(X.shape[0])
        for i, observation in enumerate(X):
            predictions[i] = self.single_prediction(observation, self.root)
        return predictions


class RegressionTree(DecisionTree):
    """
    Implementation of a regression tree

    Attributes
    ----------
    cost : cost function for tree
    max_depth : maximum depth of tree
    min_size : minimum size of the data being split
    min_impurity : minimum percent of homogenous data required for the building
    of certain branch of tree to be halted
    min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
    """

    def __init__(self, max_depth=None, min_size=5, min_cost=0, in_forest=False):
        """
        Parameters
        ----------
        max_depth : maximum depth of tree
        min_size : minimum size of the data being split
        min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
        in_forest : specifies whether tree will be a part of a random forest
        """

        self.cost = 'mse'
        self.max_depth = max_depth
        self.min_size = min_size
        self.min_cost = min_cost
        self.in_forest = in_forest
        super().__init__(
            cost_func=self.cost,
            max_depth=self.max_depth,
            min_size=self.min_size,
            min_impurity=None,
            min_cost=self.min_cost,
            in_forest=self.in_forest)


class ClassificationTree(DecisionTree):
    """
    Implementation of a classification tree

    Attributes
    ----------
    cost : cost function for tree
    max_depth : maximum depth of tree
    min_size : minimum size of the data being split
    min_impurity : minimum percent of homogenous data required for the building
    of certain branch of tree to be halted
    min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
    """

    def __init__(self, cost_func='mcr', max_depth=None, min_size=1, min_cost=0, in_forest=False):
        """
        Parameters
        ----------
        max_depth : maximum depth of tree
        min_size : minimum size of the data being split
        min_cost : minimum cost difference i.e. the minimum amount gained from splitting data
        in_forest : specifies whether tree will be a part of a random forest
        """

        self.cost = cost_func
        self.max_depth = max_depth
        self.min_size = min_size
        self.min_cost = min_cost
        self.in_forest = in_forest
        super().__init__(
            cost_func=self.cost,
            max_depth=self.max_depth,
            min_size=self.min_size,
            min_impurity=None,
            min_cost=self.min_cost,
            in_forest=self.in_forest)


class DecisionNode():
    """
    Represents a single node in the binary decision tree that will be built

    Attributes
    ----------
    threshold : Value that determines how the data is split
    mean_dist : If the node is in a regression tree, this will be the mean of the
    values in this node. If the node is in a classification tree, this will be the
    distribution of classes in this node
    feature : the feature to split the data on based on the threshold
    type : specifies the type of node, can either be regression node or classification node
    left_child : the left child of this node in the decision tree
    right_child : the right child of this node in the decision tree
    """

    def __init__(self, threshold=None, mean_dist=None, feature=None):
        """
        Initiliazes Node using data

        Parameters
        ----------
        threshold : Value that determines how the data is split
        mean_dist : If the node is in a regression tree, this will be the mean of the
        values in this node. If the node is in a classification tree, this will be the
        distribution of classes in this node
        feature : the feature to split the data on based on the threshold
        """

        self.threshold = threshold
        self.mean_dist = mean_dist
        self.feature = feature
        self.right = None
        self.left = None
