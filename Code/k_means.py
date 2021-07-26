import numpy as np

def cal_distance(a,b):
  # a and b should be arrays
  return np.dot(a,b) / np.linalg.norm(a) / np.linalg.norm(b)
  #return euclidean_distance(a,b)

def euclidean_distance(a,b):

  return np.linalg.norm(np.array(a[:-1])-np.array(b[:-1]))

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


if __name__ == '__main__':
  #points = [[1,2],[3,4],[2,3],[100,100],[101,102]]

  with open('features.npy','rb') as f:
    features = np.load(f)
  features_norm = features / np.sum(features,axis=0)
  #print(features_norm[:,0])
  correct = 0
  index = []
  for i in range(496):
    if i % 2 == 0:
      index.append(0)
    else:
      index.append(1)
  print(len(index))
  
  features_norm = np.concatenate((features_norm.T,np.array([index]))).T
  print(features_norm.shape)
  cen_1,cen_2 = k_means(features_norm)
  #print(cen_1,cen_2)
  for i,j in zip(cen_1,cen_2):
    if i[-1] != 1:
      correct += 1
    if j[-1] == 1:
      correct += 1

  print(correct)








