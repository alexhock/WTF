__author__ = 'AlexH'

import numpy as np


def sort_array(array_data, col_to_sort, asc=True):
    sorted_indices = array_data[:, col_to_sort].argsort()
    if asc is False:
        descending_sorted_indices = sorted_indices[::-1]
        descending_sorted_cluster_counts = array_data[descending_sorted_indices]
        return descending_sorted_cluster_counts
    else:
        return array_data[sorted_indices]


def get_cluster_size_list(data_point_cluster_index):
    data_point_cluster_index = np.array(data_point_cluster_index)
    max_cluster_index = np.max(data_point_cluster_index[:, 1])
    cluster_data_points = []
    for i in range(max_cluster_index + 1):
        cluster = data_point_cluster_index[data_point_cluster_index[:, 1] == i]
        size_data_points = cluster.shape[0]
        if size_data_points > 0:
            cluster_data_points.append([i, size_data_points])
    return np.array(cluster_data_points)