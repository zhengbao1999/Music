# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def cal_distance(a,b):
  # a and b should be arrays
  #return np.dot(a,b) / np.linalg.norm(a) / np.linalg.norm(b)
  return euclidean_distance(a,b)

def euclidean_distance(a,b):

  return np.linalg.norm(np.array(a)-np.array(b))

def transition_matrix_distance(a,b):
  #compute the average euclidean distance for each row in transition matrix
  return 1


def k_means(points,luster=2):
  cen_1 = points[0]
  cen_2 = points[-1]
  change = True
  while change:
    print(cen_1,cen_2)
    cluster_1 = []
    cluster_2 = []
    #assign points to clusters
    for p in points:
      dis_1 = cal_distance(cen_1,p)
      dis_2 = cal_distance(cen_2,p)
      if dis_1 > dis_2:
        cluster_1.append(p)
      else:
        cluster_2.append(p)
    #compute new centroid
    temp = np.inf
    new_cen_1 = None

    for p in cluster_1:
      distance = 0
      for p_prime in cluster_1:
        distance += cal_distance(p, p_prime)
      if distance < temp:
        temp = distance
        new_cen_1 = p

    temp = np.inf
    new_cen_2 = None
    for p in cluster_2:
      distance = 0
      for p_prime in cluster_2:
        distance += cal_distance(p, p_prime)
      if distance < temp:
        temp = distance
        new_cen_2 = p
    if ((new_cen_1 == cen_1).all() and (new_cen_2 == cen_2).all()) or ((new_cen_1 == cen_2).all() and (new_cen_2 == cen_1).all()):
      change = False
    else:
      cen_1 = new_cen_1
      cen_2 = new_cen_2

  return cluster_1, cluster_2

pca = PCA(n_components=3)

plt.figure(figsize=(16,10))
x = [1,2,3]
y = [1,1,4]
#sns.scatterplot(x=x,y=y,hue=[2,2,3],alpha=0.3)
path = '/Users/songhewang/Downloads/Armor_acmmm2021/Music/Code/features.npy'
features = np.load(open(path,'rb'))
features_norm = features / np.sum(features,axis=0)
#pca_result = pca.fit_transform(features_norm)
#print(pca_result.shape)
#print(pca.explained_variance_ratio_)


cluster_1,cluster_2 = k_means(features_norm)

colors = [1] * len(cluster_1) + [2] * len(cluster_2)
all_feats = cluster_1+cluster_2
pca_result = pca.fit_transform(all_feats)
print(pca_result.shape)
#ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(
    xs=pca_result[:,0], 
    ys=pca_result[:,1], 
    zs=pca_result[:,2], 
    c=colors, 
    cmap='tab10'
)







