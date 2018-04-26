#utf
import data, util
import argparse
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from hopkins import hopkins
import test_util

def main(arg):
    start = time.time()
    source_data = data.Biofile(arg)
    sample = source_data.get_header()
    feature = source_data.get_index()
    X = source_data.get_matrix()
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
    if method == "knmf":
        k = input("the kernel is:")
        p = input("parameter for kernel is:")
        p = float(p)
        F, G, result = util.kernel_non_negative_factorization(X, n_components = component, max_iter=100, kernel=k, parameter=p)
    F, G = preprocessing.normalize(F, norm='l2'),np.around(preprocessing.normalize(G, norm='l2'),3)
    end = time.time()
    G_df = pd.DataFrame(G, index= sample)
    G_df.to_csv('G_{}.csv'.format(method))
    print (end - start)
    print("nmf finished, and the file of probility matrix is generated as csv in current folder")





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
       X = data.Biofile(args["hopkins"]).table
       result = hopkins(X)
       print(result)

   if args["data"]:
       main(args["data"])

   if args["draw"]:
       test_util.test_plot(args["draw"])
       #plot_heatmap_cvx(args["draw"])
       #plot_clustering_cvx(args["draw"])

   if args["test"]:
       if args["test"] == "sample":
           test_util.test_result()
       elif args["test"] == "Kmeans":
           test_util.test_kmeans_result()
       elif args["test"] == "draw":
           test_util.test_draw()
       elif args["test"] == "kernel":
           test_util.test_kernel()
       else:
           raise SyntaxError("needed test from: sample, Kmeans, draw, kernel")
