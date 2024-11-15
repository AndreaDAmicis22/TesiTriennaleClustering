from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin

class FuzzyCMeansSizeInsensitive(BaseEstimator,ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, n_init=10, metric='euclidean', m=2, method='fuzzy'):
    self.n_clusters = n_clusters        ##c - [2,n]
    self.epsilon = epsilon              ##epsilon
    self.iters = iters                  ##tau - n° iterations
    self.random_state = random_state    ##??
    self.n_init = n_init
    self.metric = metric
    self.m = m                          ##m - degree of fuzziness
    self.method = method

  def fit(self, X, y=None):
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters)
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))
    self.cluster_assignments = np.random.rand(X.shape[0], self.n_clusters)
    self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]

    for it in range(self.iters):
      dists = pairwise_distances(X, self.centroids, metric=self.metric)

      #calcolo di quanti elementi contiene ogni cluster e salvataggio nell'array a
      a = np.zeros(self.n_clusters)
      i_s = np.zeros(X.shape[0])
      for i in range(X.shape[0]):
        max = 0
        coordinate_max = 0
        for j in range(self.n_clusters):
          if(self.cluster_assignments[i,j] > max):
            max = self.cluster_assignments[i,j]
            coordinate_max = j
        i_s[i] = coordinate_max
        a[coordinate_max] += 1

      #calcolo S_i
      s = np.zeros(self.n_clusters)
      for i in range(self.n_clusters):
        s[i] = a[i] / X.shape[0]   
    
      #calcolo di p_j
      self.ro = np.zeros(X.shape[0])
      for i in range(X.shape[0]):
        self.ro[i] = (1-s[int(i_s[i])])/(np.min(s)+self.epsilon)
      
      #calcolo grado di membership fuzzy
      for i in range(X.shape[0]):
        for j in range(self.n_clusters):
          num = dists[i,j]
          val = 0
          denom = 0
          for k in range(self.n_clusters):
            denom += dists[i,k]
            val += (num/(denom+self.epsilon))**(1/(self.m-1))
          self.cluster_assignments[i,j] = self.ro[i]*(1/(val+self.epsilon))
    
      #ricalcolo dei centroidi
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m*X, axis=0)/np.sum(self.cluster_assignments[:,k]**self.m)
    return self

  def predict(self, X):
    dists = pairwise_distances(X, self.centroids, metric=self.metric)

    #calcolo di quanti elementi contiene ogni cluster e salvataggio nell'array a
    a = np.zeros(self.n_clusters)
    i_s = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      max = 0
      coordinate_max = 0
      for j in range(self.n_clusters):
        if(self.cluster_assignments[i,j] > max):
          max = self.cluster_assignments[i,j]
          coordinate_max = j
      i_s[i] = coordinate_max
      a[coordinate_max] += 1

    #calcolo S_i
    s = np.zeros(self.n_clusters)
    for i in range(self.n_clusters):
      s[i] = a[i] / X.shape[0]   
    
    #calcolo di p_j
    self.ro = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      self.ro[i] = (1-s[int(i_s[i])])/(np.min(s)+self.epsilon)

    #calcolo grado di membership fuzzy
    for i in range(X.shape[0]):
      for j in range(self.n_clusters):
        num = dists[i,j]
        val = 0
        denom = 0
        for k in range(self.n_clusters):
          denom += dists[i,k]
          val += (num/(denom+self.epsilon))**(1/(self.m-1))
        self.cluster_assignments[i,j] = self.ro[i]*(1/(val+self.epsilon))

    if self.method == 'fuzzy':
      self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    elif self.method == 'possibility':
      self.cluster_assignments = self.cluster_assignments/np.max(self.cluster_assignments, axis=1)[:, np.newaxis]
    return self.cluster_assignments

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)