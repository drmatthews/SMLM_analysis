import argparse
import math
import os
import time
from math import ceil
import csv

import numpy as np
import pandas as pd
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


X_COLUMN = 'x'
Y_COLUMN = 'y'
Z_COLUMN = 'z'
WX_COLUMN = 'wx'
WY_COLUMN = 'wy'
UNCERTAINTY_COLUMN = 'uncertainty'
FRAME_COLUMN = 'frame'
INTENSITY_COLUMN = 'intensity'
DETECTIONS_COLUMN = 'detections'

COLUMNS = [X_COLUMN, Y_COLUMN, Z_COLUMN,
           WX_COLUMN, WY_COLUMN,
           UNCERTAINTY_COLUMN, FRAME_COLUMN,
           INTENSITY_COLUMN, DETECTIONS_COLUMN]

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


class LocsHDFStore:

    """
    Note - A localisation is a single row with all columns
    """
    def __init__(self, filepath=None, mode='read', **kwargs):
        if filepath.endswith('h5'):
            self.filepath = filepath
            filename = os.path.basename(filepath)
            self.basename = os.path.splitext(filename)[0]

            if 'read' in mode:
                if os.path.exists(filepath):
                    store = pd.HDFStore(filepath, mode='r+', **kwargs)
                    self._check_keys(store)
                    metadata = store.get_storer(self._key).attrs.metadata
                    self.n_locs = int(metadata['n_locs'])
                    self.n_frames = int(metadata['n_frames'])
                    self.source = metadata['source']
                    self.metadata = metadata
                    self.store = store
            elif 'write' in mode:
                self.store = pd.HDFStore(filepath, mode='w', **kwargs)
                self._key = None  

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.store.close()

    def close(self):
        self.store.close()

    def _check_keys(self, store):
        keys = store.keys()
        if '/linked_table' in keys:
            self._key = '/linked_table'
        elif '/table'in keys:
            self._key = '/table'
        else:
            raise IOError('no data in this store')
        
        tracks_key = '/tracks_table'
        if tracks_key in keys:
            self.tracks_key = tracks_key
        else:
            self.tracks_key = None

    def rename_table(self, key, new_key):
        """
        Give a table a new name. Use with caution.
        """
        keys = self.store.keys()
        if key in keys:
            self.store.get_node(key)._f_rename(new_key)             
        else:
            raise IOError('Table does not exist in file')

    def remove_table(self, key):
        """
        Remove a table from the store. Use with caution.
        """
        keys = self.store.keys()
        if key in keys:
            self.store.remove(key)
        else:
            raise IOError('Table does not exist in file')

    @property
    def table(self):
        return self.store.get(self.key)

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        keys = self.store.keys()
        if value not in keys:
            self._key = value

    @property
    def frames(self):
        """
        A sorted list of integer frame numbers

        Return
        ------

        list(int)
        """
        frame_nos = self.store.select_column(self.key, FRAME_COLUMN).unique()
        frame_nos.sort()
        return frame_nos

    def get_localisations(self):
        """
        Return all rows - for
        use if the DataFrame is not in memory

        Returns
        -------
        DataFrame
        """
        df = self.store.get(self.key)
        return df

    def get_coords(self):
        """
        Return all the (x,y,z) coordinates.

        Returns
        -------
        np.array
        """
        df = self.store.select(self.key, columns=[X_COLUMN, Y_COLUMN, Z_COLUMN])
        x = df[X_COLUMN].values
        y = df[Y_COLUMN].values
        z = df[Z_COLUMN].values
        return np.array((x, y, z)).T

    def get(self, frame_no, start=None, stop=None, columns=None):
        """
        Get the set of localisations with a 
        specfied frame number (integer)

        Parameters
        ----------
        frame_no (int) - the movie frame number
        start (int) - input to HDFStore.select() - start selection at this row
        stop (int) - input to HDFStore.select() - stop selection at this row
        columns (list) - the columns to select from the table

        Returns
        -------
        DataFrame
        """
        if start and stop:
            assert start < stop
        frame = self.store.select(
            self.key, '{0} == {1}'.format(FRAME_COLUMN, frame_no),
            start=start, stop=stop, columns=columns
        )
        return frame

    def get_in_frame_range(self, start=None, stop=None, columns=None):

        """
        Gets the localisations in the specified movie frame range

        Returns
        -------
        DataFrame
        """

        assert start < stop
        frames = self.store.select(
            self.key, '{0} > {1} & {0} < {2}'.format(
                FRAME_COLUMN, start, stop
            ), columns=columns
        )
        return frames

    def get_localisations_block(self, start=None, stop=None, columns=None):
        """
        Get the set of localisations with a 
        specfied table row number range (integer)

        Returns
        -------
        DataFrame
        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.n_locs - 1

        assert start < stop
        block = self.store.select(
            self.key, start=start, 
            stop=stop, columns=columns
        )
        return block

    def get_table_rows(self, row_indices, key=None):
        if key is None:
            key = self.key

        return self.store.select(key, where=row_indices)

    def get_localisations_in_frame(self, frame_no):
        """
        Get the localisations in the specfied frame

        Returns
        -------
        DataFrame
        """
        frame = self.get(frame_no)
        return frame 

    def get_coords_in_frame(self, frame_no):
        """
        Get the (x,y,z) coordinates of the localisations
        in the specfied frame

        Returns
        -------
        np.array
        """
        cols = [X_COLUMN, Y_COLUMN, Z_COLUMN]
        try:
            frame = self.get(frame_no, columns=cols)
            x = frame[X_COLUMN].values
            y = frame[Y_COLUMN].values
            z = frame[Y_COLUMN].values
            return np.array((x, y, z)).T
        except:
            return np.array([])

    def get_store_keys(self):
        return self.store.keys()

    def get_xyz(self):
        """
        Return all the (x,y,z) coordinates.

        Returns
        -------
        Pandas DataFrame
        """
        df = self.store.select(
            self.key, 
            columns=[X_COLUMN, Y_COLUMN, Z_COLUMN, FRAME_COLUMN]
        )
        return df.reset_index(drop=True) 

    def get_width(self):
        df = self.store.select(
            self.key, columns=[WX_COLUMN, WY_COLUMN, FRAME_COLUMN]
        )
        return df.reset_index(drop=True)

    def get_width_mean(self):
        df = self.store.select(
            self.key, columns=[WX_COLUMN, WY_COLUMN, FRAME_COLUMN]
        )
        mean_df = df.groupby(FRAME_COLUMN).mean()
        return mean_df

    def get_width_variance(self):
        df = self.store.select(
            self.key, columns=[WX_COLUMN, WY_COLUMN, FRAME_COLUMN]
        )
        var_df = df.groupby(FRAME_COLUMN).var()
        return var_df

    def locs_frame_iterator(self, start=None, stop=None, skip_empty=False):
        """
        A generator of localisations - gets all localisations in a
        movie frame.

        Returns
        -------
        Dataframe
        """
        if start and stop:
            assert start < stop

        if skip_empty:
            frames = self.frames
        else:
            if start is None:
                start = int(self.frames[0])
            if stop is None:
                stop = int(self.frames[-1])
            frames = list(range(start, stop + 1))
        if start == stop:
            stop = start + 1

        selected = [f for f in frames if f >= start and f <= stop]
        for frame_no in selected:
            yield self.get(frame_no)

    def coords_iterator(self, start=None, stop=None):
        """
        A generator of (x,y,z) coordinates

        Returns
        -------
        tuple (frame_no, points)
        """
        frames = self.frames()
        if start is None:
            start = frames[0]
        if stop is None:
            stop = frames[-1]

        if start == stop:
            stop = start + 1

        assert start < stop
        selected = [f for f in frames if f >= start and f < stop]
        cols = [X_COLUMN, Y_COLUMN, Z_COLUMN]
        for frame_no in selected:
            df = self.get(frame_no, columns=cols)
            x = df[X_COLUMN].values
            y = df[Y_COLUMN].values
            z = df[Y_COLUMN].values
            yield frame_no, np.array((x, y, z)).T

    def _area_under_curve(self, x, y):
        return abs(np.trapz(y, x))

    def adjacent_dist_mat(self):
        """
        
        """
        fmax = 5000
        frames = self._table.groupby('frame')
        nn = []
        for qf, qframe in frames:
            if qf < fmax:
                x = qframe[X_COLUMN].values
                y = qframe[Y_COLUMN].values
                q_points = np.array((x, y)).T
                for adj_f, adj_frame in frames:
                    if adj_f < fmax:
                        x = adj_frame[X_COLUMN].values
                        y = adj_frame[Y_COLUMN].values
                        adj_points = np.array((x, y)).T
                        kdt = KDTree(adj_points, leaf_size=30, metric='euclidean')
                        if adj_points.shape[0] == 1:
                            k = 1
                            nn_id = 0
                        else:
                            k = 2
                            nn_id = 1
                        adj_dist, _ = kdt.query(q_points, k=k, return_distance=True)
                        nn.append(adj_dist[:, nn_id])

        nn_adj = np.concatenate(nn)
        nn_adj[nn_adj > 200] = 200.0
        hist, edges = np.histogram(nn_adj, bins=99, range=(1,199), normed=False)
        centers = (edges[:-1] + edges[1:]) / 2
        area = self._area_under_curve(centers, hist)
        print(area)

        # a1, a2, a3, sig, w, xc
        params = [area / 2, area / 2, hist[98] / 200.0, 10.0, 201.0, 100.0]
        curve, results = curve_fitting.fit_adj_nn(hist, centers, params)
        return (hist, centers, curve)

    def clear_locs(self):
        keys = self.store.keys()
        for key in keys:
            if self.key in key:
                self.store.remove(key)
        
        self.key = None

    def write_locs(self, locs, key=None):
        """
        Iteratively write localisations to HDF5
        """
        if key is None:
            key = '/table'

        try:
            df = locs
            if not isinstance(locs, pd.DataFrame):
                df = pd.DataFrame(locs)
                df.columns = COLUMNS

            self.store.append(
                key, df, format='t',
                append=True, data_columns=True
            )
        except:
            raise IOError('Could not write localisations to file')

    def clear_links(self):
        keys = self.store.keys()
        for key in keys:
            if '/linked_table' in key:
                self.store.remove(key)
            if '/tracks_table' in key:
                self.store.remove(key)
    
        self.tracks_key = None
        self._key = '/table'

    def write_tracks(self, tracks):
        """
        Iteratively write tracks to table in HDF5
        """        
        self.tracks_key = '/tracks_table'
        try:
            df = pd.DataFrame(tracks)
            self.store.append(
                self.tracks_key, df, format='t',
                append=True, data_columns=True
            )
            self.store.get_storer(self.tracks_key).attrs.metadata = self.metadata
        except:
            raise IOError('Could not write tracks to file')

    def write_track(self, track):
        """
        Write a single track as a separate table
        """
        key = '/track_{}'.format(track['track_id'].values[0])
        # print(key)
        try:
            self.store.put(key, track, format='t', data_columns=True)
        except:
            raise IOError('Could not write track to file')

    def get_tracks(self):
        if self.tracks_key:
            columns = ['frame', 'track_id']
            tracks = self.store.select(
                self.tracks_key, columns=columns
            )
            tracks.index = range(len(tracks.index))
            return tracks
        else:
            raise IOError('No tracks in file')

    def track_iterator(self, num_tracks):
        if self.tracks_key:
            for track_id in range(num_tracks):
                track = self.store.select(
                    self.tracks_key,
                    '{0} == {1}'.format('track_id', track_id)
                )
                yield track
        else:
            raise IOError('No tracks in file')

    def write_metadata(self, metadata, key=None):
        if key is None:
            key = '/table'
        self.store.get_storer(key).attrs.metadata = metadata

    def write_to_malk(self):
        fname = self.basename + '_malk.txt'
        fpath = os.path.join('.\\test_data', fname)
        print(fpath)
        out_file = open(fpath, "w")
        out_file.write('# localizations'+"\n")
        out_file.close
        out_file = open(fpath, "a")        
        fmax = self.n_frames
        for l, loc in self._table.iterrows():
            if loc[FRAME_COLUMN] < fmax:
                loc_list = [loc[X_COLUMN], loc[Y_COLUMN], loc[Z_COLUMN], loc[FRAME_COLUMN], loc[INTENSITY_COLUMN]]
                loc_str = [str(i) for i in loc_list]
                row = ' '.join(loc_str) + '\n'
                out_file.write(row)
            else:
                break
        out_file.close()


class ClustersHDFStore(LocsHDFStore):
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
