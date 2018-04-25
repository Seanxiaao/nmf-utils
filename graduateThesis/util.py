import numbers
import math

import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state, check_array
import _update_snmf_fast as up

EPSILON = np.finfo(np.float32).eps
INTEGER_TYPES = (numbers.Integral, np.integer)

"""
this is the codes for utils of main.py
"""



def get_lst_location(lst, index, reverse = False):
    """ get the location of input, if input is name, output is location,
    if input is location, reverse should set be True and the output
    is name. """
    if (not reverse):
        for i, item in enumerate(lst):
            if item == index:
                return i
    else:
        for i, item in enumerate(lst):
            if i == index:
                return item


def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError('Array passed to %s is full of zeros.' % whom)

def get_pos_value(value):
     return (math.fabs(value) + value) / 2

def M_pos(a):
     row = a.shape[0]
     column = a.shape[1]
     for i in range(row):
         for j in range(column):
              a[i, j] = get_pos_value(a[i,j])
     return a

def losses(X, F, G):
    return np.linalg.norm(X - np.dot(F,G.T))


def kernel_M(X, initialization, parameter):
    shape, Xt = X.shape[1], X.T
    result = np.zeros([shape, shape])
    for i in range(shape):
        for j in range(shape):
            if initialization == 'rbf':
                result[i, j] = rbf_kernel(Xt[i], Xt[j], parameter)
            elif initialization == 'poly':
                result[i, j] = pol_kernel(Xt[i], Xt[j], parameter)
            elif initialization == 'sigmoid':
                result[i, j] = sig_kernel(x, y, parameter)
    return result

#kernels we needed
def rbf_kernel(x, y, sigma): #x and y are vectors
    return pow(2.718, - sigma * np.linalg.norm(x - y) ** 2)

def pol_kernel(x, y, dimension):
    return (1 + np.dot(x.T, y)) ** dimension

def sig_kernel(x, y, alpha):
    return np.tanh(alpha * np.dot(x.T, y))

#----quality function ----
def sparscity(X):
    n = X.shape[0] * X.shape[1]
    return (n - (np.linalg.norm(X, ord = 1)/np.linalg.norm(X, ord = 2)) ** 2)/(n - 1)

def accuracy(X):
    pass

#---- algorithm ----
#we can check it's sensitiveness to initials and try transfer learning
# {X - FG.T}
def semi_non_negative_factorization(X, F=None, G=None, n_components = None,
                                    tol=1e-4,max_iter=200, initialization = "random"):
    X = check_array(X, accept_sparse=('csr', 'csc'))
    #if non componets is inputted, the components is initialized to n
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    #check initialized position
    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)
    #initialized G
    #_check_init(H, (n_components, n_features), "NMF (input H)")
    #_check_init(W, (n_samples, n_components), "NMF (input W)")
    if initialization == "Kmeans":
        F = np.zeros((n_samples, n_components))
        Gt = np.zeros((n_components, n_features)) #G has been transposed here
        E = np.ones((n_components, n_features))
    #W = {W_{ik} = 1 if x_{i} belongs to cluster k otherwise 0}

        kmeans = KMeans(n_clusters = n_components, random_state = 0).fit(X.T)
        labels = kmeans.labels_
    #test should be fine
        for i, label in enumerate(labels):
            Gt[label][i] = 1
        Gt += 0.2 * E
        G = Gt.T
    elif initialization == "random" :   #alternative method
        G = np.random.rand(n_features, n_components)
    F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))

    #initialize F
    #F = X*W*(W^{T}W)^{-1} X.shape = (row, column)
    #update, can use cython to boost(max_iter + 1, n_components, n_features
    result = []
    for n_iter in range(max_iter):
    #  alternative method
    #    F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))
    #    XtF, FtF = np.dot(X.T, F), np.dot(F.T, F)
    #    XtF_p,  FtF_p = M_pos(XtF), M_pos(FtF)
    #    XtF_n,  FtF_n = XtF_p - XtF, FtF_p - FtF
    #    numerator, denominator = XtF_p + np.dot(G, FtF_n), XtF_n + np.dot(G, FtF_p)
    #    G *= np.sqrt(numerator / denominator)

        G = up._update_snmf_fast(X, F, G, 0.000001)
        F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))
        #print("F:{} \n G:{}".format(preprocessing.normalize(F,norm='l2'),preprocessing.normalize(G,norm='l2')))
        if (n_iter % 1) == 0:
            los = losses(X, F, G)
            print("It is {} times iteration for semi_NMF with losses :{} ".format(n_iter, los))
            result.append(los)
    return F, G, result
    ## write some simple test now to check the mathods value

def semi_non_negative_factorization_with_straint(X, F=None, G=None, n_components = None,
                                    alpha = 0, beta = 0, tol=1e-4,max_iter=200, initialization = "random"):
    X = check_array(X, accept_sparse=('csr', 'csc'))
    #if non componets is inputted, the components is initialized to n
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    #check initialized position
    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)
    #initialized G
    #_check_init(H, (n_components, n_features), "NMF (input H)")
    #_check_init(W, (n_samples, n_components), "NMF (input W)")
    if initialization == "Kmeans":
        F = np.zeros((n_samples, n_components))
        Gt = np.zeros((n_components, n_features)) #G has been transposed here
        E = np.ones((n_components, n_features))
    #W = {W_{ik} = 1 if x_{i} belongs to cluster k otherwise 0}

        kmeans = KMeans(n_clusters = n_components, random_state = 0).fit(X.T)
        labels = kmeans.labels_
    #test should be fine
        for i, label in enumerate(labels):
            Gt[label][i] = 1
        Gt += 0.2 * E
        G = Gt.T
    elif initialization == "random" :   #alternative method
        G = np.random.rand(n_features, n_components)
    F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))

    #initialize F
    #F = X*W*(W^{T}W)^{-1} X.shape = (row, column)
    #update, can use cython to boost(max_iter + 1, n_components, n_features
    result , Ir = [], np.identity(n_components)
    for n_iter in range(max_iter):
        G = up._update_snmf_fast_constraint(X, F, G, 0.000001, beta)
        F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G) + alpha * Ir)))
        #print("F:{} \n G:{}".format(preprocessing.normalize(F,norm='l2'),preprocessing.normalize(G,norm='l2')))
        if (n_iter % 1) == 0:
            los = losses(X, F, G)
            print("It is {} times iteration for semi_NMF({}, {}) with losses :{} ".format(n_iter, alpha, beta, los))
            result.append(los)
    return F, G, result
    ## write some simple test now to check the mathods value


def convex_non_negative_factorization(X, F=None, G=None, n_components = None,
                                    tol=1e-4,max_iter=200, initialization = "random"):
    X = check_array(X, accept_sparse=('csr', 'csc'))
    #if non componets is inputted, the components is initialized to n
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    #check initialized position
    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    Ht = np.zeros((n_components, n_features)) #G has been transposed here
    E = np.ones((n_components, n_features))
    #W = {W_{ik} = 1 if x_{i} belongs to cluster k otherwise 0}
    kmeans = KMeans(n_clusters = n_components, random_state = 0).fit(X.T)
    labels = kmeans.labels_
    #test should be fine
    for i, label in enumerate(labels):
        Ht[label][i] = 1
    D_inv = np.diag([1/sum(row) for row in Ht])
    Gt = Ht + 0.2 * E
    #initialized G0 and W0
    G, W = Gt.T, np.dot(Gt.T, D_inv)
    XtX = np.dot(X.T,X)
    XtX_p = M_pos(XtX)
    XtX_n = XtX_p - np.dot(X.T, X)
    result = []
    for n_iter in range(max_iter):
        W, G = up._update_cvxnmf_fast(X, W, G, XtX, XtX_p, XtX_n)
    #    #update G alternation
    #    XtX = np.dot(X.T,X)
    #    XtX_p = M_pos(XtX)
    #    XtX_n = XtX_p - np.dot(X.T, X)
    #    numerator_g = np.dot(XtX_p, W) + np.dot(G,np.dot(W.T, np.dot(XtX_n, W)))
    #    denominator_g = np.dot(XtX_n, W) + np.dot(G,np.dot(W.T, np.dot(XtX_p, W)))
    #    for label in range(n_components):
    #        for i in range(n_features):
    #            G[i][label] *= math.sqrt(numerator_g[i, label]/ denominator_g[i, label])

    #    WGtG = np.dot(W,np.dot(G.T,G))
    #    #update W
    #    numerator_W  = np.dot(XtX_p, G) + np.dot(XtX_n,WGtG)
    #    denominator_W = np.dot(XtX_n,G) + np.dot(XtX_p,WGtG)
    #    for label in range(n_components):
    #        for i in range(n_features):
    #            W[i][label] = math.sqrt(numerator_W[i, label]/ denominator_W[i, label])

        if (n_iter % 1) == 0:
            los = losses(X,np.dot(X,W),G)
            print("It is {} times iteration for convex-NMF with losses {} and sparseness:".format(n_iter, los))
            result.append(los)
    return W, G, result

def kernel_non_negative_factorization(X, F=None, G=None, n_components = None,
                                    tol=1e-4,max_iter=200, kernel = 'rbf', parameter = 0.5):

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    #check initialized position
    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    K, Ir = kernel_M(X, kernel, parameter), np.eye(n_features)
    F = np.random.rand(n_features, n_components) # G is sensitive for initial position
    Ht = np.zeros((n_components, n_features))
    E = np.ones((n_components, n_features))
    kmeans = KMeans(n_clusters = n_components, random_state = 0).fit(X.T)
    labels = kmeans.labels_
    #test should be fine
    for i, label in enumerate(labels):
        Ht[label][i] = 1
    D_inv = np.diag([1/sum(row) for row in Ht])
    Gt = Ht + 0.2 * E
    #initialized G0 and W0
    G = Gt
    result = []
    for n_iter in range(max_iter):
        W = Ir - np.dot(F, G)
        D = np.dot(W.T, np.dot(K, W)) ** (-1/2)
        temp_lst = []
        for i in range(len(D)):
            temp_lst.append(D[i, i])
        D = np.diag([x for x in temp_lst])
        numerator, denominator =np.array(np.dot(K,np.dot(D, G.T))), \
                                np.array(np.dot(K,np.dot(F, np.dot(G, np.dot(D,G.T)))))
        #tempF = np.dot(K,np.dot(D, np.dot(G.T * np.linalg.inv(np.dot(K,np.dot(F, np.dot(G, np.dot(D,G.T))))))))
        for i in range(n_features):
            for j in range(n_components):
                if denominator[i, j] == 0:
                    pass
                else:
                    F[i, j] *= numerator[i, j]/denominator[i, j]

        numerator, denominator = np.dot(D, np.dot(K, F)), np.dot(D, np.dot(G.T,np.dot( F.T ,np.dot( K , F))))
        #tempG = np.dot(D, np.dot(K ,np.dot(F , np.linalg.inv(np.dot(D , np.dot( G.T ,np.dot( F.T ,np.dot( K , F))))))))
        for i in range(n_components):
            for j in range(n_features):
                if denominator[j, i] == 0:
                    pass
                else:
                    G[i, j] *= numerator[j, i] / denominator[j, i]
        if (n_iter % 1) == 0:
            los = losses(X,np.dot(X,F),G.T)
            print("It is {} times iteration for kernel-NMF with losses {} and sparseness:".format(n_iter, los))
            result.append([n_iter,los])


    return F, G, result


#(BaseEstimator, TransformerMixin)
class semi_NMF:
    """semi-Non-Negative Matrix Factorization (NMF)

    Find two matrices (W, H) whose product approximates the
    matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.
    """

    def __init__(self, n_components=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))

        W, H, n_iter_ = non_negative_factorization(
            X=X, W=W, H=H, n_components=self.n_components, init=self.init,
            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        self.reconstruction_err_ = _beta_divergence(X, W, H, self.beta_loss,
                                                    square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data
        """
        check_is_fitted(self, 'n_components_')

        W, _, n_iter_ = non_negative_factorization(
            X=X, W=None, H=self.components_, n_components=self.n_components_,
            init=self.init, update_H=False, solver=self.solver,
            beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            alpha=self.alpha, l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        return W

    def inverse_transform(self, W):
        """Transform data back to its original space.

        Parameters
        ----------
        W : {array-like, sparse matrix}, shape (n_samples, n_components)
            Transformed data matrix

        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix of original shape

        .. versionadded:: 0.18
        """
        check_is_fitted(self, 'n_components_')
        return np.dot(W, self.components_)
