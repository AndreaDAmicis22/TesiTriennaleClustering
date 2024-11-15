from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

import _fuzzyclustering as fuzzy
import RFCM as r_fuzzy
import KFCM as k_fuzzy
import CFCM as c_fuzzy
import csiFCM as csi_fuzzy

X, t = load_iris(return_X_y=True)

#print(t)
#print(X.shape) => (150,4)
#dataset = dataset.drop(columns=["Model"])

x = fuzzy.FuzzyCMeans(n_clusters = 3)
y = r_fuzzy.FuzzyCMeansRobusted(n_clusters = 3)
z = k_fuzzy.FuzzyCMeansKernelized(n_clusters = 3)
w = c_fuzzy.FuzzyCMeansCredibilistic(n_clusters = 3)
k = csi_fuzzy.FuzzyCMeansSizeInsensitive(n_clusters = 3)

#z.fit(X)
#z.fit_predict(X)
a = k.fit_predict(X)
print(a)
print("\n")