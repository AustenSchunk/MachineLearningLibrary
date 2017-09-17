import numpy as np
from sklearn.utils.extmath import safe_sparse_dot as safedot




class Kernel():
    """
    Class contains various kernels that will be used in place of a dot
    product between two vectors. These kernel functions can be thought
    of as a measure of similarity between two vectors
    """

    def gaussian(self, x, x_, **kwargs):
        """
        Computes the Gaussian(RBF) kernel of the two given vectors

        Parameters
        ----------
        x : D x 1 feature vector
        x_ : D x 1 feature vector
        sigma : parameter controlling the bandwith of the kernel

        Returns
        -------
        kernelized inner product of the two given vectors
        """
        if kwargs:
            try:
                sigma = kwargs['sigma']
            except KeyError:
                raise ValueError('Must use proper parameters of Gaussian(RBF) kernel')
        else:
            sigma = 30

        gamma = 1 / (2 * np.square(sigma))
        sq_norm = safedot(x, x) + safedot(x_,x_) - (2 * safedot(x, x_))
        n = gamma * sq_norm
        return np.exp(-n)

    def polynomial(self, x, x_, **kwargs):
        """
        Computes the polynomial kernel of the two given vectors

        Parameters
        ----------
        x : D x 1 feature vector
        x_ : D x 1 feature vector
        c : C >= 0 is a free parameter that controls the influence of higher order
        terms versus lower order terms in polynomial
        d : scaling degree

        Returns
        -------
        Kernelized inner product of the two given vectors
        """
        if kwargs:
            try:
                c = kwargs['c']
                d = kwargs['d']
            except KeyError:
                raise ValueError('Must use proper arguments for polynomial kernel')
        else:
            c = 1
            d = 2

        n = safedot(x, x_) + c
        return np.power(n, d)

    def sigmoid(self, x, x_, **kwargs):
        """
        Computes the sigmoid kernel of the two given vectors

        Parameters
        ----------
        x : D x 1 feature vector
        x_ : D x 1 feature vector
        alpha : scaling parameter 
        c = : shifting parameter
        """
        if kwargs:
            try :
                alpha = kwargs['alpha']
                c = kwargs['c']
            except KeyError :
                raise ValueError('Must use proper arguments for sigmoid kernel function.')
        else:
            alpha = 1e-10
            c = 0.5

        n = alpha * safedot(x, x_) + c
        return np.tanh(n)

