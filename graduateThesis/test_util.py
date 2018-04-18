import data, util
import numpy as np
import unittest
from sklearn.cluster import MiniBatchKMeans


class TestNMF(unittest.TestCase):
    """test cases for util, I then further implemented this one. """

    def test_result(self):
        X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                      [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                      [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                      [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                      [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                      ])

        F_expected = np.array([[0.05, 0.27],
                               [0.4, -0.4],
                               [0.7, -0.72],
                               [0.3, 0.08],
                               [-0.51, 0.49]])

        Gt_expected = np.array([[0.61, 0.89, 0.54, 0.77, 0.14, 0.36, 0.84],
                                [0.12, 0.53, 0.11, 1.03, 0.60, 0.77, 1.16]])

        F, G = util.semi_non_negative_factorization(X, n_components = 2)
        Gt = G.T
        print("F expected {}, but got {}".format(F_expected, F))
        print("G expected {}, but got {}".format(Gt_expected, Gt))

    def test_kmeans_result(self):
        X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                      [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                      [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                      [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                      [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                      ])

        C_expected = np.array([[0.29, 0.52],
                               [0.45,-0.32],
                               [0.59,-0.60],
                               [0.46,0.36],
                               [-0.41, 0.37]])

        estimator = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100,
                     batch_size=100, verbose=0, compute_labels=True,
                     random_state=None, tol=0.0, max_no_improvement=10,
                     init_size=None, n_init=3, reassignment_ratio=0.01).fit(X.T)
        C_factual = estimator.cluster_centers_
        C_factual = preprocessing.normalize(C_factual, norm='l2')
        print("C expected \n {},\n but got \n {}".format(C_expected, C_factual.T))

    def test_kmeans_label(self):
        X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                      [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                      [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                      [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                      [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                      ])

        expected = [0, 0, 0, 1, 1, 1, 1]
        estimator = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100,
                     batch_size=100, verbose=0, compute_labels=True,
                     random_state=None, tol=0.0, max_no_improvement=10,
                     init_size=None, n_init=3, reassignment_ratio=0.01).fit(X.T)

        factual = estimator.labels_
        print("label expected \n {}\n and got \n {}".format(expected, factual.T))


if __name__ == '__main__':
    unittest.main()
