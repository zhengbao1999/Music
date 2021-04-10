import numpy as np

def cal_distance(a,b):
  # a and b should be arrays
  return 1

def euclidean_distance(a,b):
  return 1

def transition_matrix_distance(a,b):
  #compute the average euclidean distance for each row in transition matrix
  return 1


def k_means(points.cluster=2):
  cen_1 = points[0]
  cen_2 = points[-1]
  change = True
  while change:
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
    if new_cen_1 == cen_1 and new_cen_2 == cen_2:
      change = False
    else:
      cen_1 = new_cen_1
      cen_2 = new_cen_2




