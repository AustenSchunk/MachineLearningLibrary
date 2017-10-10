import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import safe_sparse_dot as safedot
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

class LinearClassificationModel():

    """
    This module includes the following classification models:
    Binary:
        -- Perceptron and Logistic Regression
    Multi-class:
        -- Logistic Regression

    Methods used to solve logistic regression problem:
    Binary:
        -- Iteratively reweighted least squares
    Both Binary and Multi-Class:
        -- Batch stochastic gradient descent
        -- Newton conjugate gradient
        -- Broyden–Fletcher–Goldfarb–Shanno algorithm

    Note: All the formulations of these models can be found in the following books  
    "Machine Learning: A Probabilistic Perspective" by Kevin Murphy
    "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright
    """

    def __init__(self, binary = False):
        """
        Initializes Linear Classification Model

        Parameters
        ----------
        binary : whether class labels are binary or not
        """
        self.binary = binary


    def step_size(self, iter_num, k = 0.5, t = 100):
        """
        Computes the size of step to be taking in direction of descent of a given function.
        This is based largely on the current iteration of the iterative algorithm being used
        to minimize given function 

        Parameters
        ----------
        iter_num : number of current iteration being used in algorithm
        k : must be in range (0,1) and larger values will decrease step sizes more
        t : larger values will decrease the step size taken

        Returns
        -------
        step size determined by this algorithm

        Note
        ----
        The choice of step size search is not arbitrary, but another step size
        can algorithm can easily replace this one.

        """
        return ((t + iter_num) ** (-k))


    def threshold(self, y, threshold):
        """
        Used for binary thresholding of class probabilities to class labels

        Parameters
        ----------
        y : predicted probabilities for each data observation to be a 0 or a 1
        threshold : some value in the range (0, 1) that decides the decision boundary
        of what class each observation belongs to

        Returns
        -------
        y : bit vector containing 0 or 1 depending on which class each belongs to
        """
        y[y >= threshold] = 1
        y[y < threshold] = 0
        return y


    def convert_labels_binary(self,X, w, threshold = 0.5):
        """
        Converts values to 0 or 1 based on thresholding of expit function

        Parameters
        ----------
        X : N x D matrix composed of numerical features
        w : Weights determined by chosen solver
        of what class each observation belongs to
        threshold : Some value in the range (0, 1) that decides the decision boundary
        of what class each observation belongs to

        Returns
        -------
        y: bit vector containing 0 or 1 depending on which class each belongs to

        """
        n_pred = expit(safedot(X, w))
        y = self.threshold(n_pred, threshold)
        return y


    def score_irls(self, y, pi, S):
        """
        Gives the working response after each update it irls algorithm

        Parameters
        ----------
        y : Actual label for each feature vector
        pi : Prediction from [0,1], where closer proximity to 0 or 1 means it is more likely for
        that feature vector to be either a 0 or 1 respectively
        S : Diagonal matrix calculating from the decomposition of the Hessian

        Returns
        -------
        error : response of predictions compared to actual values
        """
        error = safedot(la.pinv(S),(y - pi))
        return error


    def irls(self, X, y, iterations = 50, l2_reg = 0, save_weights = False):
        """
        Solving the likelihood function using Iteratively Reweighted Least Squares method

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        max_iters : maximum number of iterations before the algorithm will terminate
        l2_reg : amount of l2 regularization to be applied
        save_weights : whether or not the weights obtain from each iteration should be saved

        """

        w = np.zeros(X.shape[1])
        prediction = np.ones(X.shape[1])
        if save_weights:
            W = np.empty(shape = ((iterations), w.shape[0]))
        i = 0
        while i < iterations:
            n = safedot(X,w)
            prediction = expit(n) # predicting new y values based on current weights
            s = prediction * (1 - prediction)
            s[s == 0] = np.finfo('float').resolution # Ensuring that matrix S will be invertible
            S = np.diag(s)

            z = n + self.score_irls(y, prediction, S) # response variable
            if np.allclose(z, n):
                if (save_weights):
                    W = W[:i]
                break
            w_0 = la.inv(la.multi_dot((X.T,S, X)))
            w_1 = la.multi_dot((X.T, S, z))

            w_n = safedot(w_0, w_1)
            w = w_n + (l2_reg * w)
            if save_weights:
                W[i] = w
            i += 1
        self.w = w
        if save_weights:
            self.W = W


    def init_bfgs(self, X, y):
        """
        Initializes the BFGS algorithm

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0

        Returns
        -------
        W : initial weights
        B : 2 x D x D "matrix", where the first is the initial pseudo-Hessian matrix and
        the second is empty
        G : Gradient matrix containing the initial gradient vector and an empty slot for the next
        gradient vector

        """
        W = np.zeros(shape = (2, X.shape[1])) # initializing weights 
        B = np.zeros(shape = (2, X.shape[1], X.shape[1]))

        B[0] = np.diag(np.ones(shape=X.shape[1])) # Initializing pseudo-hessian to identity matrix
        pi = expit(safedot(X, W[0]))

        G = np.empty(shape = (2, X.shape[1]))
        G[0] = safedot(X.T,(pi - y))
        return W, B, G


    def update_B(self, B, W, G):
        """
        Performs the update of the pseudo-Hessian Matrix

        Parameters
        ----------
        B : 2 x D x D "martix" containing pseudo-Hessian matrix and an empty slot for the next one
        W : 2 x D matrix containing the two weights calculated from the past two iterations of
        the BFGS algorithm
        G : 2 x D matrix containing the two gradients calculated from the past two iterations
        of the BFGS algorithm

        Returns
        -------
        B : 2 x D x D matrix containing past two pseudo hessian matrices
        """

        dW = W[1] - W[0]
        dG = G[1] - G[0]
        t_1a = safedot(dG, dG)
        t_1b = safedot(dG, dW)
        if not t_1b:
            t_1 = 0
        else: 
            t_1 = t_1a / t_1b
        t_2a = safedot(safedot(B[0], dW), safedot(B[0], dW))
        t_2b = safedot(safedot(dW, B[0]),dW)
        B[1] = B[0] + t_1 + (t_2a / t_2b)
        return B[1]


    def bfgs(self, X, y, iterations = 20, save_weights = False, t = 0,  k = .12):

        """
        Implementation of the Broyden–Fletcher–Goldfarb–Shanno algorithm

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        max_iters : maximum number of iterations before the algorithm will terminate
        save_weights : whether or not the weights obtain from each iteration should be saved
        k : must be in range (0,1) and larger values will decrease step sizes more
        t : larger values will decrease the step size taken

        Note
        ----
        Significantly faster convergence with normalized X
        """
        W, B, G = self.init_bfgs(X,y)
        weights = np.empty(shape = (iterations, W.shape[1]))

        for i in range(iterations):     
            n = safedot(X,W[0])
            G[1] = (safedot(X.T, expit(n) - y)  ) / G.shape[1]
            d = safedot(la.pinv(B[0]), G[1])

            a = self.step_size(iter_num = (i + 1) * 10 , t = t, k = k)

            W[1] = W[0] - (a * d)
        
            B[0] = self.update_B(B, W, G)
            
            G[0] = G[1]
            W[0] = W[1]

            if save_weights:
                weights[i] = W[0]
            
        self.w = W[0] 
        if save_weights:
            self.W = weights


    def get_batch(self, X, y, iteration, size):
        """
        Used in batch gradient descent to attain the current batch of data

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        iteration : current iteration in batch sgd algorithm
        size : size of the batch of data to be returned

        Returns
        -------
        batch : batch of data that will be used in batch sgd algorithm
        labels: bit vector of corresponding to class labels of the batch of data
        """
        max_idx = iteration * size % X.shape[0]
        min_idx = max_idx - size % X.shape[0]
        if max_idx < size:
            batch = np.concatenate((X[min_idx:], X[:max_idx]))
            labels = np.concatenate((y[min_idx:], y[:max_idx]))
        else:
            batch = X[min_idx:max_idx]
            labels = y[min_idx: max_idx]
        return batch, labels


    def batch_sgd(self, X, y, epsilon, max_iters = 100, size = 100, save_weights = False):
        """
        Uses batch stochastic gradient descent to determine weights used in prediction of outputs

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        epsilon : small number to be used to test for approximate convergence
        max_iters : maximum number of iterations before the algorithm will terminate
        size : determines the size of each batch of data
        save_weights : whether or not the weights obtain from each iteration should be saved
        """
        w = w_0 = np.zeros(X.shape[1])
        if save_weights:
            W = np.zeros(shape=(max_iters, X.shape[1]))
        i = 0
        while i < max_iters:
            batch_x, batch_y = self.get_batch(X, y, i, size)
            pi = expit(np.dot(batch_x, w))
            gradient = np.dot(batch_x.T,(pi - batch_y))
            alpha = self.step_size((i + 1) * size)
            
            w = w_0 - (alpha * gradient)
            if save_weights:
                W[i] = w

            if la.norm(w_0 - w) < epsilon:
                if save_weights:
                    W = W[:i]
                break
            w_0 = w
            i += 1
        if save_weights:
            self.W = W
        self.w = w


    def conjugate_gradient(self, A, b, epsilon, x = None):
        """
        Solves linear equation Ax = b using conjugate gradients. This Algorithm
        can be found on pg. 111 of "Numerical Analysis"

        Parameters
        ----------
        A : In the context of newton methods, A is the Hessian matrix
        b : In the context of newton methods, b is the gradient vector
        epsilon : Since covergence to 0 would be a waste of computational resources, epsilon is used
        to determine if the residual is "close-enough" to 0
        x : can be given if there is a desired starting 

        Returns
        -------
        x : approximate solution to the linear equation of the form Ax = b
        """
        if x is None:
            x = np.ones(b.shape[0]) # Initialize random x
        r_0 = safedot(A, x) - b # Calculating residual 
        p = - r_0 # original direction

        while la.norm(r_0) > epsilon:
            alpha = safedot(r_0,r_0) / safedot(safedot(p,A),p) # Creating 1-D minimizer

            x = x + (alpha * p) # Updating x
            r_1 = r_0 + (alpha * safedot(A, p)) # Updating residual
            beta = safedot(r_1, r_1) / safedot(r_0,r_0) # matrix to ensure conjugacy between new direct p and A

            p = -r_1 + safedot(beta, p) # updating direction
            r_0 = r_1
        return x


    def binary_newton_cg(self, X, y, max_iters = 10, l2_reg = 0, save_weights = False, epsilon = 1e-4):
        """
        Solves the optimization of the likelihood function using newton's 
        method with conjugate gradients

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        max_iters : maximum number of iterations before the algorithm will terminate
        l2_reg : determines the amount of regularization that will be applied at each iteration
        save_weights : whether or not the weights obtain from each iteration should be saved
        epsilon : small number to be used to test for approximate convergence when determining
        the direction to be descended
        """
        w = np.zeros(X.shape[1])
        if save_weights:
            W = np.zeros(shape=(max_iters, X.shape[1]))
        for i in range(max_iters):

            mu = expit(safedot(X,w)) # Calculating predicted probabilities

            g = safedot(X.T,(mu - y)) # gradient vector
            H = safedot(safedot(X.T, np.diag(mu)),X) # hessian matrix

            n = self.conjugate_gradient(H, g, epsilon) # 

            w = (w - n) + (l2_reg * w) # updating weights with l2 regularization 
            if save_weights:
                W[i] = w
        if save_weights:
            self.W = W # weights of each iteration
        self.w = w # final weights


    def binary_classify(self, X):
        """
        Performs binary classification of feature vector X

        Parameters
        ----------
        X : M x D matrix composed of numerical features, where each feature is 1 x D
        """
        self.predictions = self.convert_labels_binary(X, self.w)


    def fit(self, X, y, solver = 'newton_cg', max_iters = 10, l2_reg = 0.5, save_weights = False, convergence_param = 1e-4):
        """
        Fits model based using one of the solvers listed at the class documentation
        
        Parameters
        ----------
        X : N x D matrix composed of numerical features
        y : either N X 1 or N X M, where M is the number of classes. If N X 1, y is a bit vector corresponding
        to whether each of the N features is in class 1 or 0. If N X M each of the N row vectors is a bit vector, 
        where one bit will be turned on and that will correspond to its class
        solver : method of minimizing the likelihood function
        max_iters : maximum number of iterations to be used in the minimization of the likelihood function
        l2_reg : determines the amount of regularization that will be applied at each iteration
        save_weights : whether or not the weights obtain from each iteration should be saved
        convergence_param : parameter used to approximate 0
        """

        if solver not in ['newton_cg', 'batch_sgd', 'perceptron', 'irls', 'bfgs']:
            raise Exception("Must choose proper solver")

        if (np.unique(y).shape[0]) > 2:
            self.binary = False
            if solver not in ['newton_cg', 'batch_sgd']:
                raise Exception("Must use proper solver for multi-class data")
            else:
                self.model = 'Multi-Class'
                self.solver = solver
                pass
        else: 
            self.binary = True;

            self.model = 'Binary'
            self.solver = solver

            if (solver == 'newton_cg'):
                self.binary_newton_cg(X, y, max_iters, l2_reg, save_weights = save_weights, epsilon = convergence_param)

            elif (solver == 'batch_sgd'):
                self.batch_sgd(X, y, save_weights = save_weights, max_iters = max_iters, epsilon = convergence_param)

            elif (solver == 'irls'):
                self.irls(X, y, max_iters, l2_reg = l2_reg, save_weights = save_weights)

            elif (solver == 'bfgs'):
               self.bfgs(X, y, iterations = max_iters, save_weights = save_weights)


    def classify(self, X):
        """
        Classifies data uses weights calculated by fit() method and new matrix X of validation or test weights

        Parameters
        ----------
        X : M x D matrix composed of numerical features, where each feature is 1 x D
        """
        if self.binary:
            self.binary_classify(X)


class Perceptron():
    """
    Implementation of Rosenblatt's Perceptron Algorithm
    """

    def fit(self, X, y, max_iters = 10000, save_weights = False, epsilon = 1e-5):
        """
        Finds optimal weights by adjusting them only when incorrect observations are made

        Parameters
        ----------
        X : N x D matrix composed of numerical features, where each feature is 1 x D
        y : N X 1 matrix, where y is a bit vector corresponding to whether each of the
        N features is in class 1 or 0
        max_iters : maximum number of iterations before the algorithm will terminate
        save_weights : whether or not the weights obtain from each iteration should be saved
        epsilon : small number to be used to test for approximate convergence when determining
        the direction to be descended
        """
        if not np.allclose(np.unique(y), np.array([-1,1])):
            y[y == 0] = -1

        w = - np.ones(shape=X.shape[1]) + np.finfo('float').resolution
        W = np.zeros(shape = (max_iters, w.shape[0]))
        s = np.zeros(w.shape[0])
        for i in range(max_iters):
            idx = i % (X.shape[0] -1)
            x = X[idx]
            y_hat = np.sign(safedot(x, w))

            # Only updates when an incorrect predict is made
            if not np.allclose(y_hat, y[idx]):
                w += y[idx] * x
            
            W[i] = w
            if i > 50 and la.norm(W[i] - W[i - 50]) < epsilon:
                break
        self.w = w
        self.W = W


    def classify(self, X):
        """
        Uses weights calculating from fitting to form classification of new features

        Parameters
        ----------
        X : M x D matrix composed of numerical features, where each feature is 1 x D
        """
        preds = np.sign(safedot(X,self.w))
        preds[preds < 1] = 0
        self.predictions = preds


class VisualizeLinearModel():

    """
    Creates visualizations for the learning curves of the above linear models
    """
    def __init__(self, style = 'seaborn-darkgrid'):
        """
        Initializes the visualization model

        Parameters
        ----------
        style : the style of plot that will be used in future visualizations
        """
        plt.style.use(style)
        sns.set()


    def prepare_visualization(self, model, X_test, y_test, X_train, y_train):
        """
        Prepares the data needed in order to visualize learning rates and the confusion matrix

        Parameters
        ----------
        model : linear model that was used to find the weights needed to predict y based on X
        X_test: matrix of input values that will be weight in order to attempt to predict y_test
        y_test : actual output that the model will attempt to predict using X_test
        X_train: matrix of input values from the testing data matrix, used in similar manner to X_test
        y_train : output that the model will attempt to predict using X_train

        Note
        ----
        Must have the save_weights paramter set to True when fitting the linear model in order for this
        method to work.
        """
        size = model.W.shape[0];
        train_accuracy = np.zeros(shape=(size))
        test_accuracy = np.zeros(shape= train_accuracy.shape)

        confusion_matrix_test = np.zeros(shape = (size, 2, 2))
        confusion_matrix_train = np.zeros(shape = (size, 2, 2))

        for i in range(size):
            train_predictions = model.convert_labels_binary(X_train, model.W[i])
            test_predictions = model.convert_labels_binary(X_test, model.W[i])
            train_accuracy[i] = np.sum(train_predictions == y_train) / y_train.shape[0]
            test_accuracy[i] = np.sum(test_predictions == y_test) / y_test.shape[0]
            confusion_matrix_train[i] = metrics.confusion_matrix(y_train, train_predictions)
            confusion_matrix_test[i] = metrics.confusion_matrix(y_test, test_predictions)

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy

        self.train_TN = confusion_matrix_train[:,0,0] / np.sum(y_train == 0)
        self.train_FN = confusion_matrix_train[:,1,0] / np.sum(y_train == 0)
        self.train_TP = confusion_matrix_train[:,1,1] / np.sum(y_train == 1)
        self.train_FP = confusion_matrix_train[:,0,1] / np.sum(y_train == 1)

        self.test_TN = confusion_matrix_test[:,0,0] / np.sum(y_test == 0)
        self.test_FN = confusion_matrix_test[:,1,0] / np.sum(y_test == 0)
        self.test_TP = confusion_matrix_test[:,1,1] / np.sum(y_test == 1)
        self.test_FP = confusion_matrix_test[:,0,1] / np.sum(y_test == 1)


    def show_visualization(self, line_size):
        """
        Displays data that was calculated in prepare visualization

        Parameters
        ----------
        line_size : thickness of lines in plots
        """
        fig1, (ax1, ax2) = plt.subplots(nrows = 2, figsize = (9, 7))

        ax1.set_title("Training Results")
        ax1.plot(self.train_TN, linewidth = line_size, alpha = 0.7, label = "True Negative")
        ax1.plot(self.train_FN, linewidth = line_size, alpha = 0.7, label = "False Negative")
        ax1.plot(self.train_TP, linewidth = line_size, alpha = 0.7, label = "True Positive")
        ax1.plot(self.train_FP, linewidth = line_size, alpha = 0.7, label = "False Positive")
        ax1.legend(frameon=True, loc=5, ncol=1)

        ax2.set_title("Testing Results")
        ax2.plot(self.test_TN, linewidth = line_size, alpha = 0.7, label = "True Negative")
        ax2.plot(self.test_FN, linewidth = line_size, alpha = 0.7, label = "False Negative")
        ax2.plot(self.test_TP, linewidth = line_size, alpha = 0.7, label = "True Positive")
        ax2.plot(self.test_FP, linewidth = line_size, alpha = 0.7, label = "False Positive")
        ax2.legend(frameon=True, loc=5 , ncol=1)

        fig2, ax3 = plt.subplots()
        ax3.set_title("Error Rates")
        ax3.plot(1 - self.train_accuracy, linewidth = line_size, alpha = 0.8,  label = "Training Error")
        ax3.plot(1 -self.test_accuracy, linewidth = line_size, alpha = 0.8, label = "Testing Error")
        ax3.legend(frameon=True, ncol=2);
        plt.show()


    def visualize(self, model, X_test, y_test, X_train, y_train, line_size = 1):
        """
        Calcualtes and displays the learning rate and confusion matrix

        Parameters
        ----------
        model : linear model that was used to find the weights needed to predict y based on X
        X_test: matrix of input values that will be weight in order to attempt to predict y_test
        y_test : actual output that the model will attempt to predict using X_test
        X_train: matrix of input values from the testing data matrix, used in similar manner to X_test
        y_train : output that the model will attempt to predict using X_train
        line_size : thickness of lines in plots

        Note
        ----
        Must have the save_weights paramter set to True when fitting the linear model in order for this
        method to work.
        """
        self.prepare_visualization(model, X_test, y_test, X_train, y_train)
        self.show_visualization(line_size)

