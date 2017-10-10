import numpy as np
from sklearn.utils.extmath import safe_sparse_dot as safedot
from ..Utilities.utils import Kernel

# import os
# print(os.getcwd())
# import sys
# sys.exit(1)

class NearestNeighbors():
    """
    This class implements the k-nearest neighbors algorithm using
    both kernelized and standard distance metrics
    """


    def l1_distance(self, X, x):
        """
        Computes the Manhattan distance between a single feature observation from testing
        data and each observation from training data

        Parameters
        ----------
        X : N x D matrix consisting of N observations of data
        x : D x 1 vector consisting of a single observation

        Returns
        -------
        N x 1 vector of distances between x and each observation in X
        """

        raw_dist = np.abs(X - x)
        return np.sum(raw_dist, axis = 1)


    def l2_distance(self, X, x, gen_dist, kernel, **kwargs):
        """
        Computes Euclidean distance between a single feature observation from testing
        data and each observation from training data

        Parameters
        ----------
        X : N x D matrix consisting of N observations of data
        x : D x 1 vector consisting of a single observation
        gen_dist : pre computed squared norm of each row vector in X
        kernel : function that will apply a pseudo projection of the features into a space
        of different dimensions

        Returns
        -------
        N x 1 vector of distances between x and each observation in X
        """

        if kernel is None:
            t_1 = gen_dist
            t_2 = safedot(x, x)
            t_3 = 2 * safedot(X, x)
            return t_1 + t_2 - t_3
        else:
            distances = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                t_2 = kernel(x, x, **kwargs)
                t_3 = kernel(X[i], x, **kwargs)
                distances[i] = gen_dist[i] + t_2 - (2 * t_3)
            return distances


    def lp_distance(self, X, x, p = 3):
        """
        Computes the Minkowski distance between a single feature observation from testing
        data and each observation from training data

        Parameters
        ----------
        X : N x D matrix consisting of N observations of data
        x : D x 1 vector consisting of a single observation

        Returns
        -------
        N x 1 vector of distances between x and each observation in X
        """

        abs_dist = np.abs(X - x)
        nth_dist = np.sum(np.power(abs_dist, p), axis = 1)
        distances = np.power(nth_dist, (1 / p))
        return distances


    def min_distances(self, distances, y, k):
        """
        Given the distance between each training observation and a given
        testing observation, computes the predicted label for the testing observation based
        off the median of label of the k nearest neighbors

        Parameters
        ----------
        distances : N x 1 vector of distances between each training observation and a
        certain testing observation
        y : labels of each training observation
        k : number of nearest neighbors to compare to the testing observation

        Returns
        -------
        median of the labels of the k nearest neighbors. If there is a tie, k will be
        decreased by one until the die is broken
        """

        while k > 0:
            min_distances = np.argpartition(distances, k)[:k]
            labels = y[min_distances]
            prediction = np.median(labels)
            if prediction.is_integer() or k == 1:
                return prediction
            else:
                k -= 1
        return prediction


    def classify(self, X_train, y_train, X_test, k = 3, dist_func = "l2", p = 3, kernel = None, **kwargs):
        """
        Classifies a given matrix or feature vector of testing data by assigning
        each new feature a label based on the labels of the k nearest neighbors

        Parameters
        ----------
        X_train : N x D matrix consisting of N observations of training data
        y_train : N x 1 vector consisting of N labels, where each corresponds to 
        the feature vector of X_train at the same index
        X_test : M x D vector consisting of a M observations of testing data
        k : number of nearest neighbors to compare to each testing observation
        dist_func : metric that determines distance between two vectors
        p : parameter used in l3 distance function
        kernel : function that will apply a pseudo projection of the features into a space
        of different dimensions
        kwargs : parameters specific to kernelized k nearest neighbors. See utils for more information

        Returns
        -------
        predictions of the labels for each feature vector in X_test

        Note
        ----
        The use of a kernel will only work if the given distance function is l2_distance 
        """

        func_map = {'l1': self.l1_distance, 'l2': self.l2_distance, 'lp' : self.lp_distance}

        kern = Kernel()
        kernel_map = {'rbf': kern.gaussian, 'polynomial' : kern.polynomial, 'sigmoid' : kern.sigmoid}

        if dist_func == 'l2':
            gen_dist = np.sum(X_train * X_train, axis = 1)

        if kernel:
            gen_dist = np.zeros(X_train.shape[0])
            for i in range(X_train.shape[0]):
                gen_dist[i] = kernel_map[kernel](X_train[i], X_train[i], **kwargs)

        pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            if dist_func == 'l2':
                if kernel == None:
                    distances = func_map[dist_func](X_train, X_test[i], gen_dist, kernel)

                else:
                    distances = func_map[dist_func](X_train, X_test[i], gen_dist, kernel_map[kernel] , **kwargs)

            elif dist_func == 'lp':
                distances = func_map[dist_func](X_train, X_test[i], p = p)

            else :
                distances = func_map[dist_func](X_train, X_test[i])

            pred[i] = self.min_distances(distances, y_train, k)
        return pred


