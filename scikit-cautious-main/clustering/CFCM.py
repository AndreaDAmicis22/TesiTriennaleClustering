from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

class FuzzyCMeansCredibilistic(BaseEstimator,ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, n_init=10, metric='euclidean', m=2, method='fuzzy', gamma=0.5):
    self.n_clusters = n_clusters        ##c - [2,n]
    self.epsilon = epsilon              ##epsilon
    self.iters = iters                  ##tau - n° iterations
    self.random_state = random_state    ##??
    self.n_init = n_init
    self.metric = metric
    self.m = m                          ##m - degree of fuzziness
    self.method = method
    self.gamma = gamma                  ##gamma - [0,1]

  def fit(self, X, y=None):
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters)
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))

    #calcolo di q
    q = int(np.ceil((X.shape[0]/self.n_clusters)*self.gamma))
    q_real = q+1

    #calcolo del q-vicinato per ogni punto dato
    neigh = NearestNeighbors(n_neighbors=q_real)
    neigh.fit(X)
    dists_n, idx = neigh.kneighbors(X)     

    #calcolo di theta_j
    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      num = 0
      for j in range(1, q+1, 1):
        num += dists_n[i,j]  
      theta[i] = num/q

    #calcolo di p_j
    ro = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      ro[i] = 1-((theta[i]-np.min(theta))/(np.max(theta)-np.min(theta)+self.epsilon))

    for it in range(self.iters):                                            
      dists = pairwise_distances(X, self.centroids, metric=self.metric) 

      #calcolo grado di membership fuzzy
      for i in range(X.shape[0]):
        for j in range(self.n_clusters):
          num = dists[i,j]
          val = 0
          denom = 0
          for k in range(self.n_clusters):
            denom += dists[i,k]
            val += (num/(denom+self.epsilon))**(1/(self.m-1))
          self.cluster_assignments[i,j] = ro[i]*(1/(val+self.epsilon))
    
      #ricalcolo dei centroidi
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m*X, axis=0)/(np.sum(self.cluster_assignments[:,k]**self.m + self.epsilon))
    return self

  def predict(self, X):
    dists = pairwise_distances(X, self.centroids, metric=self.metric)

    #calcolo di q
    q = int(np.ceil((X.shape[0]/self.n_clusters)*self.gamma))
    q_real = q+1

    #calcolo del q-vicinato per ogni punto dato
    neigh = NearestNeighbors(n_neighbors=q_real)
    neigh.fit(X)
    dists_n, idx = neigh.kneighbors(X)     

    #calcolo di theta_j
    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      num = 0
      for j in range(1, q+1, 1):
        num += dists_n[i,j]  
      theta[i] = num/q

    #calcolo di p_j
    ro = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      ro[i] = 1-((theta[i]-np.min(theta))/(np.max(theta)-np.min(theta)+self.epsilon))

    #calcolo grado di membership fuzzy
    for i in range(X.shape[0]):
      for j in range(self.n_clusters):
        num = dists[i,j]
        val = 0
        denom = 0
        for k in range(self.n_clusters):
          denom += dists[i,k]
          val += (num/(denom+self.epsilon))**(1/(self.m-1))
        self.cluster_assignments[i,j] = ro[i]*(1/(val+self.epsilon))

    if self.method == 'fuzzy':
      self.cluster_assignments = self.cluster_assignments/(np.sum(self.cluster_assignments, axis=1)[:, np.newaxis])
    elif self.method == 'possibility':
      self.cluster_assignments = self.cluster_assignments/np.max(self.cluster_assignments, axis=1)[:, np.newaxis]
    return self.cluster_assignments

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)