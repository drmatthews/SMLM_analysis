# distutils: language = c++

cimport cython
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

def voronoi_inner(int n_locs, object[::1] nlist, double[::1] density, double thresh_density, double min_samples):
    cdef int cluster_id = 1
    cdef int c, clust, i, n, v
    cdef vector[np.npy_intp] cluster
    cdef vector[np.npy_intp] queue

    cdef int[::1] labels = np.full(n_locs, -1, dtype=np.int)
    cdef int[::1] checked = np.zeros(n_locs, dtype=np.int)

    for i in range(n_locs):

        # determine whether localisation at index i has been checked already
        if checked[i] == 1:
            continue

        # if it hasn't check if it's density is greater than the threshold
        if density[i] > thresh_density:
            # mark this localisation as having been checked
            checked[i] = 1
            # add it to the current cluster
            cluster.push_back(i)
            # create a queue and put localisation neighbours in it
            queue = nlist[i]
            # print("queue {}".format(queue))
            while queue.size() > 0:
                # get an element in the queue
                n = queue.back()
                # remove it from the queue
                queue.pop_back()
                # mark it as checked
                checked[n] = 1
                if density[n] > thresh_density:
                    # add it to the current cluster
                    cluster.push_back(n)
                    # get the neighbours of this localisation and add to the queue
                    # queue += [idx for idx in nlist[n] if checked[idx] == 0]
                    neighb = nlist[n]
                    for i in range(nlist[n].shape[0]):
                        v = neighb[i]
                        if checked[v] == 0:
                            queue.push_back(v)                    
                # print(queue)

            if cluster.size() > min_samples:
                for c in range(cluster.size()):
                    clust = cluster[c]
                    labels[clust] = cluster_id
            
            cluster.clear()
            cluster_id += 1

    return np.asarray(labels)