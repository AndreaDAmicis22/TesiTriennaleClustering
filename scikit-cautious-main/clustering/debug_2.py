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

dataset = pd.read_csv("C:/Users/Andrea/Documents/Unimib/3°anno/IA/scikit-cautious-main/dataset/optdigits.CSV")
data = dataset.iloc[:, :64]
clas = dataset.iloc[:, 63]
data, clas = resample(data, clas, n_samples=500, replace=True, stratify=clas)
data = data.reset_index().iloc[:, 1:]
clas = clas.reset_index().iloc[:, 1]   

x = fuzzy.FuzzyCMeans(n_clusters = 3)
y = r_fuzzy.FuzzyCMeansRobusted(n_clusters = 3)
z = k_fuzzy.FuzzyCMeansKernelized(n_clusters = 3)
w = c_fuzzy.FuzzyCMeansCredibilistic(n_clusters = 3)
k = csi_fuzzy.FuzzyCMeansSizeInsensitive(n_clusters = 10)

a = k.fit_predict(data.to_numpy())
print(a)
print("\n")
#X, t = load_iris(return_X_y=True)
#print(t)
#print(X.shape) => (150,4)
#dataset = dataset.drop(columns=["Model"])

#z.fit(X)
#z.fit_predict(X)
#print(k.fit_predict(X))
