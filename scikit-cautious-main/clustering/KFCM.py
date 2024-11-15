from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin

class FuzzyCMeansKernelized(BaseEstimator,ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, n_init=10, metric='euclidean', m=2, method='fuzzy'):
    self.n_clusters = n_clusters        #c - [2,n]
    self.epsilon = epsilon              #epsilon
    self.iters = iters                  #tau - n° iterations
    self.random_state = random_state    #??
    self.n_init = n_init
    self.metric = metric
    self.m = m                          #m - degree of fuzziness
    self.method = method

  def fit(self, X, y=None):
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters)    
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))    

    v_mean = np.mean(X) 
    d_mean = (np.sum(np.sqrt(np.power(X-v_mean, 2))))/X.shape[0]
    sigma = (np.sum(np.power(np.sqrt(np.power(X-v_mean, 2))-d_mean, 2)))/(X.shape[0]-1)

    for it in range(self.iters):       #repeat tau times
      dists = pairwise_distances(X, self.centroids, metric=self.metric)
      dists = self.gaussianKernel(dists, sigma)

      #calcolo gradi di membership fuzzy
      for i in range(X.shape[0]):
        for j in range(self.n_clusters):
          num = dists[i,j]
          val = 0
          denom = 0
          for k in range(self.n_clusters):
            denom += dists[i,k]
            val += ((1-num)/(1-denom+self.epsilon))**(1/(self.m-1))
          self.cluster_assignments[i,j] = 1/(val+self.epsilon)

      #ricalcolo dei centroidi
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m*X*dists[:,k][:,np.newaxis], axis=0)/np.sum(self.cluster_assignments[:,k]**self.m*dists[:,k])
    return self

  def predict(self, X):
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    v_mean = np.mean(X) 
    d_mean = (np.sum(np.sqrt(np.power(X-v_mean, 2))))/X.shape[0]
    sigma = (np.sum(np.power(np.sqrt(np.power(X-v_mean, 2))-d_mean, 2)))/(X.shape[0]-1)
    dists = self.gaussianKernel(dists, sigma)
    
    #calcolo gradi di membership fuzzy
    for i in range(X.shape[0]):
      for j in range(self.n_clusters):
        num = dists[i,j]
        val = 0
        denom = 0
        for k in range(self.n_clusters):
          denom += dists[i,k]
          val += ((1-num)/(1-denom+self.epsilon))**(1/(self.m-1))
        self.cluster_assignments[i,j] = 1/(val+self.epsilon)

    if self.method == 'fuzzy':
      self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    elif self.method == 'possibility':
      self.cluster_assignments = self.cluster_assignments/np.max(self.cluster_assignments, axis=1)[:, np.newaxis]
    return self.cluster_assignments

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)
  
  def gaussianKernel(self, dists, sigma):
    return np.exp(-(dists/sigma))
    