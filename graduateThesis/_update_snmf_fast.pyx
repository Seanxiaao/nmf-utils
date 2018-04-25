# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Sean Xiao
# License: BSD 3 clause

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.math cimport sqrt

cdef double get_pos_value(double value):
     return (fabs(value) + value) / 2

cdef np.ndarray[np.double_t, ndim=2] M_pos(np.ndarray[np.double_t, ndim=2] a):
     cdef int row = a.shape[0]
     cdef int column = a.shape[1]
     for i in range(row):
         for j in range(column):
              a[i, j] = get_pos_value(a[i,j])
     return a



def _update_snmf_fast(np.ndarray[np.double_t, ndim=2] X,
                      np.ndarray[np.double_t, ndim=2] F,
                      np.ndarray[np.double_t, ndim=2] G,
                      double resizor):

    cdef Py_ssize_t n_components = F.shape[1]
    cdef Py_ssize_t n_features = G.shape[0]
    cdef double p
    cdef np.ndarray[np.double_t, ndim=2] XtF
    cdef np.ndarray[np.double_t, ndim=2] XtF_p
    cdef np.ndarray[np.double_t, ndim=2] XtF_n
    cdef np.ndarray[np.double_t, ndim=2] FtF
    cdef np.ndarray[np.double_t, ndim=2] FtF_p
    cdef np.ndarray[np.double_t, ndim=2] FtF_n
    cdef np.ndarray[np.double_t, ndim=2] numerator
    cdef np.ndarray[np.double_t, ndim=2] denominator

    cdef np.ndarray[np.double_t, ndim=2] GFtF
    cdef np.ndarray[np.double_t, ndim=2] GFtF_n
    cdef np.ndarray[np.double_t, ndim=2] GFtF_p

    #with nogil: calculate G
    XtF, FtF = np.dot(X.T, F), np.dot(F.T, F)
    XtF_p,  FtF_p = M_pos(XtF), M_pos(FtF)
    XtF_n,  FtF_n = XtF_p - XtF, FtF_p - FtF
    GFtF = np.dot(G, FtF)
    GFtF_p = M_pos(GFtF)
    GFtF_n = GFtF_p - GFtF
    numerator, denominator = XtF_p + np.dot(G, FtF_n), XtF_n + np.dot(G, FtF_p)
    #numerator, denominator = XtF_p + GFtF_n, XtF_n + GFtF_p
    for label in range(n_components):
        for i in range(n_features):
            q, p = numerator[i,label], denominator[i, label]
            if p == 0:
               pass
            #if p == 0 and q == 0:
            #   G[i][label] = 0   this block can always make the algorithm converge fast
            #elif q != 0:         which may means the result is not good
            #   G[i][label] = sqrt(numerator[i, label]/ resizor)
            else:
               G[i][label] *= sqrt(numerator[i, label] /
                                p)

    """
    XtF , GFtF = np.dot(X.T, F), np.dot(G,np.dot(F.T,F))
    for label in range(n_components):
        for i in range(n_features):
           G[i][label] = sqrt((max(0, XtF[i,label]) + max(0,GFtF[i, label])) /
                                (max(0,XtF[i,label]) + max(0, GFtF[i, label])) )
    """
    #Calculate F
    #F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))

    return G

def _update_snmf_fast_constraint(np.ndarray[np.double_t, ndim=2] X,
                          np.ndarray[np.double_t, ndim=2] F,
                          np.ndarray[np.double_t, ndim=2] G,
                          double resizor,
                          double beta):

    cdef Py_ssize_t n_components = F.shape[1]
    cdef Py_ssize_t n_features = G.shape[0]
    cdef double p
    cdef np.ndarray[np.double_t, ndim=2] Ir
    cdef np.ndarray[np.double_t, ndim=2] XtF
    cdef np.ndarray[np.double_t, ndim=2] XtF_p
    cdef np.ndarray[np.double_t, ndim=2] XtF_n
    cdef np.ndarray[np.double_t, ndim=2] FtF
    cdef np.ndarray[np.double_t, ndim=2] FtF_p
    cdef np.ndarray[np.double_t, ndim=2] FtF_n
    cdef np.ndarray[np.double_t, ndim=2] numerator
    cdef np.ndarray[np.double_t, ndim=2] denominator

    Ir = np.identity(n_components)

    #with nogil: calculate G
    XtF, FtF = np.dot(X.T, F) + beta * G, np.dot(F.T, F) + beta * Ir
    XtF_p,  FtF_p = M_pos(XtF), M_pos(FtF)
    XtF_n,  FtF_n = XtF_p - XtF, FtF_p - FtF
    numerator, denominator = XtF_p + np.dot(G, FtF_n), XtF_n + np.dot(G, FtF_p)
    for label in range(n_components):
        for i in range(n_features):
            q, p = numerator[i,label], denominator[i, label]
            if p == 0:
               #G[i][label] *= 0.001
                pass
            #if p == 0 and q == 0:
            #   G[i][label] = 0   this block can always make the algorithm converge fast
            #elif q != 0:         which may means the result is not good
            #   G[i][label] *= sqrt(numerator[i, label]/ resizor)
            else:
               G[i][label] *= sqrt(numerator[i, label] /
                                p)

    """
    XtF , GFtF = np.dot(X.T, F), np.dot(G,np.dot(F.T,F))
    for label in range(n_components):
        for i in range(n_features):
           G[i][label] = sqrt((max(0, XtF[i,label]) + max(0,GFtF[i, label])) /
                                (max(0,XtF[i,label]) + max(0, GFtF[i, label])) )
    """
    #Calculate F
    #F = np.dot(X , np.dot(G, np.linalg.inv(np.dot(G.T, G))))

    return G

def _update_cvxnmf_fast(np.ndarray[np.double_t, ndim=2] X,
                           np.ndarray[np.double_t, ndim=2] W,
                           np.ndarray[np.double_t, ndim=2] G,
                           np.ndarray[np.double_t, ndim=2] XtX,
                           np.ndarray[np.double_t, ndim=2] XtX_p,
                           np.ndarray[np.double_t, ndim=2] XtX_n
                           ):

    cdef Py_ssize_t n_components
    cdef Py_ssize_t n_features
    cdef np.ndarray[np.double_t, ndim=2] GWt
    cdef np.ndarray[np.double_t, ndim=2] WGtG
    cdef np.ndarray[np.double_t, ndim=2] numerator_g
    cdef np.ndarray[np.double_t, ndim=2] denominator_g
    cdef np.ndarray[np.double_t, ndim=2] numerator_w
    cdef np.ndarray[np.double_t, ndim=2] denominator_w

    n_components = G.shape[1]
    n_features = G.shape[0]

    #update G
    GWt = np.dot(G, W.T)
    numerator_g = np.dot(XtX_p, W) + np.dot(GWt, np.dot(XtX_n, W))
    denominator_g = np.dot(XtX_n, W) + np.dot(GWt, np.dot(XtX_p, W))

    for label in range(n_components):
        for i in range(n_features):
            G[i, label] *= sqrt(numerator_g[i, label]/ denominator_g[i, label])

    WGtG = np.dot(W,np.dot(G.T,G))

    #update W
    numerator_w  = np.dot(XtX_p, G) + np.dot(XtX_n,WGtG)
    denominator_w = np.dot(XtX_n,G) + np.dot(XtX_p,WGtG)
    for label in range(n_components):
        for i in range(n_features):
            W[i, label] *= sqrt(numerator_w[i, label]/ denominator_w[i, label])

    return W, G
