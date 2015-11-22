__author__ = 'ah14aeb'
import numpy as np
import random
from helpers import feature_helper
def calc_strength_bins(cluster_centres):

    # zscore normalise all values and sum
    #feature_helper.no

    histr, bin_edgesr = np.histogram(cluster_centres[:, 0])
    histg, bin_edgesg = np.histogram(cluster_centres[:, 5])
    histb, bin_edgesb = np.histogram(cluster_centres[:, 11])

    bin_indices = np.digitize(cluster_centres[:, 0], histr)


def random_colour_centres():

    base_path = 'F:/Users/alexh/onedrive/WTF/data/1_20_11/agglom/'
    output_folder = base_path
    cluster_centers = np.loadtxt(base_path + 'hac_cluster_centres_0.15_421.txt', delimiter='\t')

    rows = []

    for centroid_idx in range(len(cluster_centers)):

        r = int(random.random()*255)
        g = int(random.random()*255)
        b = int(random.random()*255)

        rows.append([centroid_idx, 0, r, g, b])

    np.savetxt(output_folder + '/color_activation_rule_cluster_labels_esh.txt', np.array(rows), delimiter=',', fmt="%1.1i")


def colour_centres():

    base_path = 'k:/Users/ah14aeb/ml/algos/astro/data/abell2744_nobin/20000/agglom/'
    output_folder = base_path
    cluster_centers = np.loadtxt(base_path + 'hac_cluster_centres_0.15_536.txt', delimiter='\t')

    num_steps = 25
    gradient = np.linspace(-4.0,4.0, num_steps)
    colour_step = 255/num_steps

    rows = []

    for centroid_idx in range(len(cluster_centers)):

        c = cluster_centers[centroid_idx]

        r = c[0]
        rg = c[0] + c[5]
        b = c[10]

        color = -1
        color_rgb = []
        pos = 0
        if rg > b:
        #if r > b:
            color = 0
            pos = np.digitize([rg], gradient)
            green = pos[0]*colour_step
            color_rgb = [255, green, 0]
        else:
            color = 1
            pos = np.digitize([b], gradient)
            green = pos[0]*colour_step
            color_rgb = [0, green, 255]
        if b < -2 and r < -1:
            color = 2
            color_rgb = [255, 50+(colour_step*2), 0]

        rows.append([centroid_idx, color, color_rgb[0], color_rgb[1], color_rgb[2]])

    np.savetxt(output_folder + '/color_activation_rule_cluster_labels_esh.txt', np.array(rows), delimiter=',', fmt="%1.1i")

random_colour_centres()
#colour_centres()
