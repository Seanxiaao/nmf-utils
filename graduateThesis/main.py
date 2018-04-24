#utf
import data, util
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as pyt
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from hopkins import hopkins


def main(arg):
    start = time.time()
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    X = source_data.get_matrix()
    #sample_size, feature_size = 60, 12042
    #xshape = (106 12042)
    #X = source_data.get_matrix().T[:sample_size, :feature_size]
    #X = np.array([[1.0,2.0,3.0],[-3.0,4.0,5.0],[1.0, -2.0, 3.0]])
    print("the frobenis norm of X is {}!".format(np.linalg.norm(X)))
    #clustering
    component = input("please input the number of cluster you want:")
    component = int(component)
    print("current avaiable method for nmf(should input in) is \n"
          "semi-nmf(snmf),semi-nmf with constraint(csnmf),convex-nmf(cvxnmf) and kernel-nmf(knmf)")
    method = input("input the method of nmf you want:")
    if method not in ["snmf", "csnmf", "cvxnmf", "knmf"]:
        raise ValueError("the method you put in is not available now")
    if method == "snmf":
        print("semi-nmf begins")
        F, G, result1 = util.semi_non_negative_factorization(X, n_components = component,max_iter = 100)

    if method == "csnmf":
        a = input("please input the alpha:")
        b =  input("please input in beta:")
        a, b = float(a), float(b)
        F, G, result1 = util.semi_non_negative_factorization_with_straint(X, n_components=component,
                                                                          alpha = a, beta = b, max_iter=100)
    if method == "cvxnmf":
        print("convex-nmf begins")
        F, G, result = util.convex_non_negative_factorization(X, n_components= component, max_iter=100)
    #print("\n")
    #F1, G1, result2 = util.convex_non_negative_factorization(X, n_components = group,max_iter = 100)
    if method == "knmf":
        k = input("the kernel is:")
        p = input("parameter for kernel is:")
        p = float(p)
        F, G, result = util.kernel_non_negative_factorization(X, n_components = component, max_iter=100, kernel=k, parameter=p)
    F, G = preprocessing.normalize(F, norm='l2'),np.around(preprocessing.normalize(G, norm='l2'),3)
    #F1, G1 = preprocessing.normalize(np.dot(X,F1), norm='l2'),np.around(preprocessing.normalize(G1, norm='l2'),3)
    end = time.time()
    G_df = pd.DataFrame(G, index= sample)
    #W_df, W1_df, G_df, G1_df = pd.DataFrame(F, index = sample[:sample_size]), \
    #                           pd.DataFrame(F1,index = sample[:sample_size]), \
    #                           pd.DataFrame(G, index = feature[:feature_size]),\
    #                           pd.DataFrame(G1,index = feature[:feature_size])
    #W_df.to_csv('F_semi.csv')
    #W1_df.to_csv('F1_cnvx.csv')
    G_df.to_csv('G_{}.csv'.format(method))
    #G1_df.to_csv('G1_cnvx.csv')
    print (end - start)
    print("nmf finished, and the file of probility matrix is generated as csv in current folder")
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
    #estimated = util.semi_non_negative_factorization_with_straint(X, max_iter = 10, alpha = 0.5, beta = 0.5)
    estimated = util.kernel_non_negative_factorization(X, n_components = 2, max_iter = 30, kernel= 'poly', parameter = 2)
    print("****" * 10)
    estimated1 = util.kernel_non_negative_factorization(X, n_components = 2,max_iter = 30, kernel= 'poly', parameter = 3)
    print("kernel result is {}".format(estimated[1]))
    print("kernel result is {}".format(estimated1[1]))

def test_plot(arg):

    """
    X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                  [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                  [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                  [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                  [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                  ])
    """
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 106, 12042
    sample = sample[:sample_size]
    #xshape = (106 12042)
    print(sample, feature)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    semi_r = util.semi_non_negative_factorization(X.T, max_iter = 100, n_components = 2)
    semi_r_con = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 0.5, beta = 0.5)
    semi_r_con1 = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 1, beta = 0.5)
    semi_r_con2 = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 0.5, beta = 1)
    cvx_r = util.convex_non_negative_factorization(X.T, max_iter = 60, n_components = 2)
    G, G1, G2, G4, G5 =  semi_r[1], semi_r_con[1], cvx_r[1], semi_r_con1[1], semi_r_con2[1]
    result, result1, result2, result3, result4 = semi_r[2], semi_r_con[2], cvx_r[2], semi_r_con1[2], semi_r_con1[2]
    x = [i for i in range(30)]
    # plot the losses function
    plt.title("losses function of {}".format(arg[:-4]))
    plt.xlabel("iteration times")
    plt.ylabel("losses")

    plt.plot(x, result[:30], 'r', marker = '.', label = 'sNMF')
    plt.plot(x, result1[:30], 'b', marker ='.' , label = 'con-sNMF(0.5, 0.5)')
    plt.plot(x, result3[:30], 'c', marker ='.', label = 'con-sNMF(0, 0.5)')
    plt.plot(x, result4[:30], 'm', marker ='.', label = 'con-sNMF(0.5, 1)')
    plt.plot(x, result2[:30], 'y', marker ='.' ,label = 'cvx-NMF')

    plt.legend(bbox_to_anchor=[1,1])
    plt.grid()
    plt.show()
    plt.savefig("figure1.png", dpi = 300)
    plt.close()
    #plot the clustering result
    plt1 = plt
    plt1.subplot(221)
    plt1.plot(G[:,0], G[:,1], 'ro')
    plt1.title(u'the distribution of items(sNMF)')
    #items = zip(sample, G)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(222)
    plt1.plot(G1[:,0], G1[:,1], 'bo')

    plt1.title(u'the distribution of items(sNMF(0.5, 0.5))')

    #items = zip(sample, G1)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(223)
    plt1.plot(G4[:,0], G4[:,1], 'co')
    plt1.title(u'the distribution of items(sNMF(0, 0.5))')
    #items = zip(sample, G4)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(224)
    plt1.plot(G2[:,0], G2[:,1], 'mo')
    plt1.title(u'the distribution of items(convexNMF)')
    #items = zip(sample, G2)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.show()

def test_plot_kernel(arg):
    #it is the method for test of different kernel-nmf
    """
    X = np.array([[1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
                  [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
                  [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
                  [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
                  [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
                  ])
    """
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 106, 12042
    sample = sample[:sample_size]
    #xshape = (106 12042)
    print(sample, feature)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    #semi_r = util.semi_non_negative_factorization(X.T, max_iter = 100, n_components = 2)
    #semi_r_con = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 0.5, beta = 0.5)
    #semi_r_con1 = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 1, beta = 0.5)
    #semi_r_con2 = util.semi_non_negative_factorization_with_straint(X.T, max_iter = 100, n_components = 2, alpha = 0.5, beta = 1)
    #cvx_r = util.convex_non_negative_factorization(X.T, max_iter = 60, n_components = 2)
    #G, G1, G2, G4, G5 =  semi_r[1], semi_r_con[1], cvx_r[1], semi_r_con1[1], semi_r_con2[1]
    #result, result1, result2, result3, result4 = semi_r[2], semi_r_con[2], cvx_r[2], semi_r_con1[2], semi_r_con1[2]

    x = [i for i in range(30)]
    # plot the losses function
    plt.title("losses function of {}".format(arg[:-4]))
    plt.xlabel("iteration times")
    plt.ylabel("losses")

    plt.plot(x, result[:30], 'r', marker = '.', label = 'sNMF')
    plt.plot(x, result1[:30], 'b', marker ='.' , label = 'con-sNMF(0.5, 0.5)')
    plt.plot(x, result3[:30], 'c', marker ='.', label = 'con-sNMF(0, 0.5)')
    plt.plot(x, result4[:30], 'm', marker ='.', label = 'con-sNMF(0.5, 1)')
    plt.plot(x, result2[:30], 'y', marker ='.' ,label = 'cvx-NMF')

    plt.legend(bbox_to_anchor=[1,1])
    plt.grid()
    plt.show()
    plt.savefig("figure1.png", dpi = 300)
    plt.close()
    #plot the clustering result
    plt1 = plt
    plt1.subplot(221)
    plt1.plot(G[:,0], G[:,1], 'ro')
    plt1.title(u'the distribution of items(sNMF)')
    #items = zip(sample, G)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(222)
    plt1.plot(G1[:,0], G1[:,1], 'bo')

    plt1.title(u'the distribution of items(sNMF(0.5, 0.5))')

    #items = zip(sample, G1)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(223)
    plt1.plot(G4[:,0], G4[:,1], 'co')
    plt1.title(u'the distribution of items(sNMF(0, 0.5))')
    #items = zip(sample, G4)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.subplot(224)
    plt1.plot(G2[:,0], G2[:,1], 'mo')
    plt1.title(u'the distribution of items(convexNMF)')
    #items = zip(sample, G2)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt1.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')

    plt1.show()

def plot_clustering_cvx(arg):
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 106, 12042
    #xshape = (106 12042)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    cvx_r = util.convex_non_negative_factorization(X.T, max_iter = 100, n_components = 4)
    G = cvx_r[1]
    sample = sample[:sample_size]

    plt.subplot(111)
    plt.plot(G[:,0], G[:,1], 'co')
    plt.title(u'the distribution of items(convex-nmf)')
    #items = zip(sample, G)
    #for item in items:
    #    item_name, item_data = item[0], item[1]
    #    plt.text(item_data[0], item_data[1], item_name,
    #              horizontalalignment='center',
    #              verticalalignment='top')
    plt.show()

def plot_heatmap_cvx(arg):
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    sample_size, feature_size = 106, 12042
    #xshape = (106 12042)
    X = source_data.get_matrix().T[:sample_size, :feature_size]
    cvx_r = util.semi_non_negative_factorization(X.T, max_iter=100, n_components=2)
    #cvx_r = util.convex_non_negative_factorization(X.T, max_iter = 100, n_components = 2)
    W, G = cvx_r[0], cvx_r[1]
    #print(np.dot(X.T ,W))


    sample = sample[:sample_size]
    trace = pyt.graph_objs.Heatmap(z = X,x = feature, y = sample)
    trace1 = pyt.graph_objs.Heatmap(z = W , y = feature)
    trace2 = pyt.graph_objs.Heatmap(z = G, y = sample)

    fig = pyt.tools.make_subplots(rows = 1, cols = 3,subplot_titles=('input matrix',
                                                                     'clustering matrix',
                                                                     'probility matrix'))
    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 1, 3)

    fig['layout'].update(height=1200, width=1200, title= 'HeatMap for semi-nmf')
    pyt.offline.plot(fig)

if __name__ == '__main__':
   ap = argparse.ArgumentParser()
   ap.add_argument("-d", "--data", required = False,
   help= "path to input data file")
   ap.add_argument("-t", "--test", required = False,
   help = "test started !")
   ap.add_argument("-dr", "--draw", required = False,
   help = "draw some graphs!")
   ap.add_argument("-hp", "--hopkins", required = False,
   help = "find if a set contain cluster")
   args = vars(ap.parse_args())
   if args["hopkins"]:
       hopkins(args["hopkins"])

   if args["data"]:
       main(args["data"])

   if args["draw"]:
       test_plot(args["draw"])
       #plot_heatmap_cvx(args["draw"])
       #plot_clustering_cvx(args["draw"])

   if args["test"]:
       if args["test"] == "sample":
           test_result()
       elif args["test"] == "Kmeans":
           test_kmeans_result()
       elif args["test"] == "draw":
           test_draw()
       elif args["test"] == "kernel":
           test_kernel()
       else:
           raise SyntaxError("needed test from: sample, Kmeans, draw, kernel")
