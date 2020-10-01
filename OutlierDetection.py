import numpy as np
from scipy.spatial.distance import mahalanobis



def mahalanobis_distances(numpy_arr):
    '''
    Returns the Mahalanobis distances for each example in numpy_arr. Can be used to detect outliers.
    '''
    inv_cov = np.linalg.inv(np.cov(numpy_arr.T))
    mean = np.mean(numpy_arr, axis=0)
    n = len(numpy_arr)

    distances = np.zeros(n)
    for i in range(n):
        distances[i] = mahalanobis(numpy_arr[i], mean, inv_cov)

    return distances