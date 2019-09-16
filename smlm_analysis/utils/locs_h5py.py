import argparse
import math
import os
import time
from math import ceil
import csv

import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from read_roi import read_roi_file, read_roi_zip
from natsort import natsorted

from smlm_analysis.utils import curve_fitting

NSTORM_DTYPES = {
    'Channel Name': 'category',
    'X': 'float32',
    'Y': 'float32',
    'Xc': 'float32',
    'Yc': 'float32',
    'Height': 'float32',
    'Area': 'float32',
    'Width': 'float32',
    'Phi': 'float32',
    'Ax': 'float32',
    'BG': 'float32',
    'I': 'float32',
    'Frame': 'float32',
    'Length': 'float32',
    'Link': 'float32',
    'Valid': 'float32',
    'Z': 'float32',
    'Zc': 'float32',
    'Photons': 'float32',
    'Lateral Localization Accuracy': 'float32',
    'Xw': 'float32',
    'Yw': 'float32',
    'Xwc': 'float32',
    'Ywc': 'float32'
}

THUNDERSTORM_DTYPES = {
    'frame': 'float32',
    'x [nm]': 'float32',
    'y [nm]': 'float32',
    'sigma [nm]': 'float32',
    'intensity [photon]': 'float32',
    'offset [photon]': 'float32',
    'bkgstd [photon]': 'float32',
    'uncertainty [nm]': 'float32',
    'detections': 'float32'
}


#
# helper to deal with ROIs from ImageJ
#
def get_locs_in_rois(ijroi_path, roi_scale, locs):

    if ijroi_path.endswith('zip'):
        rois = read_roi_zip(ijroi_path)
    elif ijroi_path.endswith('roi'):
        rois = read_roi_file(ijroi_path)
    else:
        raise ValueError("No ImageJ roi file exists")

    for _, roi in rois.items():
        for k, v in roi.items():
            if not isinstance(v, str):
                roi[k] = float(v) * roi_scale

        roi['locs'] = locs[
            (locs['x'] > roi['left']) &
            (locs['x'] < roi['left'] + roi['width']) &
            (locs['y'] > roi['top']) &
            (locs['y'] < roi['top'] + roi['height'])
        ].reset_index(drop=True)

    return rois


class LocsHDF:

    """
    HDF5 representation of localisations. Used for reading and
    writing. Each frame in a movie is represented in the file
    as group.
    """
    X_COLUMN = 'x'
    Y_COLUMN = 'y'
    Z_COLUMN = 'z'
    UNCERTAINTY_COLUMN = 'uncertainty'
    FRAME_COLUMN = 'frame'
    INTENSITY_COLUMN = 'intensity'
    DETECTIONS_COLUMN = 'detections'
    
    COLUMNS = [X_COLUMN, Y_COLUMN, Z_COLUMN,
               UNCERTAINTY_COLUMN, FRAME_COLUMN,
               INTENSITY_COLUMN, DETECTIONS_COLUMN]
    
    def __init__(self, filepath=None, mode='read', **kwargs):
        if filepath.endswith('h5'):
            self.filepath = filepath
            filename = os.path.basename(filepath)
            self.basename = os.path.splitext(filename)[0]

            if 'read' in mode:
                if os.path.exists(filepath):
                    store = h5py.File(filepath, mode='r')
                    self.n_locs = int(store.attrs['n_locs'])
                    self.n_frames = int(store.attrs['n_frames'])
                    self.source = store.attrs['source']
                    self.store = store
                else:
                    raise OSError('File not found.')
            elif 'write' in mode:
                self.store = h5py.File(filepath, mode='w')
        else:
            raise IOError('File not a h5 file')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.store.close()

    def close(self):
        self.store.close()

    def write_locs(self, locs):
        """
        Iteratively write localisations to HDF5.
        Each frame is written as a group. The data
        in each frame is written to a single dataset
        within the group.

        :param locs: localisations
        :type locs: numpy structured array
        """
        frame_no = int(locs[self.FRAME_COLUMN][0])

        grp = self.store.create_group('frame_{}'.format(frame_no))
        dset = grp.create_dataset('data', locs.shape, locs.dtype)

        for name in locs.dtype.names:
            dset[name] = locs[name]


class ClustersHDFStore(LocsHDF):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        # reopen the store
        self.store = pd.HDFStore(self.filepath, mode='r+')

    def add_clusters(self, cluster_id, index, labels, core_samples_mask, info):
        """
        Rewrites the table in the store to include a column of IDs
        of clusters. Writes clustering information to
        the metadata.
        """
        # remove any pre-exisiting clusters
        keys = self.store.keys()
        for key in keys:
            if cluster_id in key:
                self.store.remove(key)

        # add the clusters to a copy of the in-memory table
        # because we might be working with a subset
        # of the table we need a boolean mask of
        # which rows we are dealing with
        df = self.table
        kwargs = {'cluster_id': -1, 'core_samples': core_samples_mask}
        df = df.assign(**kwargs)
        df.loc[index, 'cluster_id'] = labels

        method = info['clustering_method']
        cluster_key = (self.basename.replace(' ', '_') +
                       '_{0}_{1}'.format(method, cluster_id))

        # write the new table to the store
        # cols = [X_COLUMN, Y_COLUMN, cluster_id]
        self.store.put(cluster_key, df, format='table', data_columns=True)

        # add the clustering info to the metadata
        metadata = {}
        metadata.update(self.metadata)
        metadata.update(info)
        print(metadata)
        # probably need a separate method for metadata update
        self.metdata = metadata
        self.store.get_storer(cluster_key).attrs.metadata = metadata

    def get_points_for_clustering(self):
        """
        Return the clustered (x,y) coordinates from
        the in-memory tables. If there aren't any clusters
        return (x,y) coordinates from the in-memory table

        Returns
        -------
        np.array
        """
        # need to adapt this to iterate over cluster tables if
        # they exist

        df = self.table
        i = df.index.values.astype(int)
        x = df[X_COLUMN].values
        y = df[Y_COLUMN].values
        return (i, np.array((x, y)).T)

    def get_cluster_points(self, method='dbscan'):
        try:
            keys = self.store.keys()
            for key in keys:
                if method in key:
                    df = self.store.get(key)
        except:
            print('No clusters found in file')

        return df[[X_COLUMN, Y_COLUMN, 'cluster_id', 'core_samples']]


if __name__=='__main__':
    locs_path = '.\\test_data\\tetra_speck_beads_time_lapse_channel_0.h5'
    lt = LocsHDFStore(locs_path)
    # print('')
    # for locs in lt.locs_iterator():
    #     print(locs.head())
    lt.close()
