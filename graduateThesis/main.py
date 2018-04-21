#utf
import data, util
import argparse
import numpy as np
import pandas as pd
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing


def main(arg):
    start = time.time()
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 60, 9000
    #xshape = (106 12042)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    #X = np.array([[1.0,2.0,3.0],[-3.0,4.0,5.0],[1.0, -2.0, 3.0]])
    print("the frobenis norm of X is {}!".format(np.linalg.norm(X)))
    #clustering
    group = 16
    F, G, result1 = util.semi_non_negative_factorization(X, n_components = group,max_iter = 100)
    print("\n")
    F1, G1, result2 = util.convex_non_negative_factorization(X, n_components = group,max_iter = 100)

    F, G = preprocessing.normalize(F, norm='l2'),np.around(preprocessing.normalize(G, norm='l2'),3)
    F1, G1 = preprocessing.normalize(np.dot(X,F1), norm='l2'),np.around(preprocessing.normalize(G1, norm='l2'),3)
    end = time.time()
    W_df, W1_df, G_df, G1_df = pd.DataFrame(F, index = sample[:sample_size]), \
                               pd.DataFrame(F1,index = sample[:sample_size]), \
                               pd.DataFrame(G, index = feature[:feature_size]),\
                               pd.DataFrame(G1,index = feature[:feature_size])
    W_df.to_csv('F_semi.csv')
    W1_df.to_csv('F1_cnvx.csv')
    G_df.to_csv('G_semi.csv')
    G1_df.to_csv('G1_cnvx.csv')
    print (end - start)
    #for row in W[:40]:
    #    print ("--" * 20)
    #    print (row)

    #have to firstly understand the data

def test_draw():
    start = time.time()
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 60, 9000
    #xshape = (106 12042)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    group = 16
    F, G, result1 = util.semi_non_negative_factorization(X, n_components = group,max_iter = 100)
    print("\n")
    F1, G1, result2 = util.semi_non_negative_factorization_with_straint(X, n_components = group, alpha = 0.5, beta = 0.5, max_iter = 100)
    print("\n")
    F2, G2, result3 = util.convex_non_negative_factorization(X, n_components = group,max_iter = 100)
    print("\n")
    F3, G3, result4 = util.kernel_non_negative_factorization(X, n_components = group,max_iter = 100)

def test_result():
    X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                  [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                  [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                  [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                  [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                  ])
    print("the Norm of X is {}".format(np.linalg.norm(X)))

    F_expected = np.array([[0.05, 0.27],
                           [0.4, -0.4],
                           [0.7, -0.72],
                           [0.3, 0.08],
                           [-0.51, 0.49]])


    Fcvx_expected = np.array([[0.31, 0.53],
                            [0.42, -0.30],
                            [0.56, -0.57],
                            [0.49, 0.41],
                            [-0.41, 0.36]])

    Gt_expected = np.array([[0.61, 0.89, 0.54, 0.77, 0.14, 0.36, 0.84],
                            [0.12, 0.53, 0.11, 1.03, 0.60, 0.77, 1.16]])

    Gcvx_expected = np.array([[0.31, 0.31, 0.29, 0.02, 0, 0,  0.02],
                              [0, 0.06, 0, 0.31, 0.27, 0.30, 0.36]])

    F, G , losses= util.semi_non_negative_factorization(X, n_components = 2)
    F1, G1, losses= util.convex_non_negative_factorization(X, n_components = 2)
    Gt = G.T
    Gt1 = G1.T
    #rescale the result

    print("\n" +"------------------" * 5 + "\n")
    F, Gt = preprocessing.normalize(F, norm='l2'),preprocessing.normalize(Gt, norm='l2')
    F1, Gt1 = preprocessing.normalize(np.dot(X,F1), norm='l2'),preprocessing.normalize(Gt1, norm='l2')
    print("Fsemi_expected:\n {} \n Fcvx_expected : \n {}  \n but got \n Fsemi :\n {} \n Fcvnx :\n {}" \
            .format(F_expected, Fcvx_expected, np.around(F,2), np.around(F1,2)))
    print("-------------------" * 5 )
    print("Gsemi_expected: \n {} \n Gcvx_expected : \n {}  \n but got \n Gsemi :\n {} \n Gcvnx : \n{}" \
            .format(Gt_expected,Gcvx_expected, np.around(Gt,2),np.around(Gt1,2)))

def test_kmeans_result():
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

def test_kmeans_label():
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

def test_kernel():

    X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                  [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                  [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                  [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                  [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                  ])
    #X = np.array([[1.0 ,0.0], [0.0, 1.0]])
    #estimated = util.semi_non_negative_factorization(X)
    estimated = util.semi_non_negative_factorization_with_straint(X, max_iter = 10, alpha = 0.5, beta = 0.5)
    #estimated2 = util.kernel_non_negative_factorization(X, kernel= 'poly', parameter = 0.5)[1]
    print("kernel result is {}".format(estimated[1]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required = False,
	help= "path to input data file")
    ap.add_argument("-t", "--test", required = False,
    help = "test started !")
    args = vars(ap.parse_args())
    if args["data"]:
        main(args["data"])

    if args["test"] == "sample":
        test_result()
    elif args["test"] == "Kmeans":
        test_kmeans_result()
    elif args["test"] == "draw":
        test_draw()
    elif args["test"] == "kernel":
        test_kernel()

    """
    nmf_model = util.NMF(n_components = 4)
    item_dis = nmf_model.fit(values)
    user_dis = nmf_model.components_
    #there should be some kinds of graphs here
    plt1 = plt
    plt1.plot(item_dis[:, 0], item_dis[:, 1], 'ro')
    plt1.draw()#直接画出矩阵，只打了点，下面对图plt1进行一些设置

    plt1.xlim((-1, 3))
    plt1.ylim((-1, 3))
    plt1.title(u'the distribution of items (NMF)')#设置图的标题

    count = 1
    zipitem = zip(item, item_dis)#把电影标题和电影的坐标联系在一起

    for item in zipitem:
        item_name = item[0]
        data = item[1]
        plt1.text(data[0], data[1], item_name,
                fontproperties=fontP,
                horizontalalignment='center',
                verticalalignment='top')
    """
