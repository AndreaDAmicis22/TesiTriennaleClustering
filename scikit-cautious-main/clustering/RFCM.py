from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin

class FuzzyCMeansRobusted(BaseEstimator,ClusterMixin):
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

    for it in range(self.iters):                                            #repeat tau times
      dists = pairwise_distances(X, self.centroids, metric=self.metric)     #euclidean distance  
      sigma = np.zeros(self.n_clusters)                                         #inizialize sigma matrix 

      #calcolo di sigma_i
      for k in range(self.n_clusters):
        sigma[k] = (np.sum((self.cluster_assignments[:,k][:,np.newaxis]**self.m)*dists)/(np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m + self.epsilon)))**(1/2)
      
      #calcolo grado di membership fuzzy
      for i in range(X.shape[0]):
        for j in range(self.n_clusters):
          num = dists[i,j]
          sigma_num = sigma[j] + self.epsilon
          val = 0
          denom = 0
          for k in range(self.n_clusters):
            denom += dists[i,k]/(sigma[k]+self.epsilon)
            val += ((num/sigma_num)/(denom+self.epsilon)+self.epsilon)**(1/(self.m-1))
          self.cluster_assignments[i,j] = 1/(val+self.epsilon)
    
      #ricalcolo dei centroidi
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m*X, axis=0)/np.sum(self.cluster_assignments[:,k]**self.m)
    return self

  def predict(self, X):
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    sigma = np.zeros(self.n_clusters)

    #calcolo di sigma_i
    for k in range(self.n_clusters):
      sigma[k] = (np.sum((self.cluster_assignments[:,k][:,np.newaxis]**self.m)*dists)/(np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m + self.epsilon)))**(1/2)
      
    #calcolo grado di membership
    for i in range(X.shape[0]):
      for j in range(self.n_clusters):
        num = dists[i,j]
        sigma_num = sigma[j] + self.epsilon
        val = 0
        denom = 0
        for k in range(self.n_clusters):
          denom += dists[i,k] / (sigma[k] + self.epsilon)
          val += ((num/sigma_num)/(denom + self.epsilon) + self.epsilon)**(1/(self.m-1))
        self.cluster_assignments[i,j] = 1/(val+self.epsilon)

    if self.method == 'fuzzy':
      self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    elif self.method == 'possibility':
      self.cluster_assignments = self.cluster_assignments/np.max(self.cluster_assignments, axis=1)[:, np.newaxis]
    return self.cluster_assignments

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)