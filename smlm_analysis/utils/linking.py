import time
from scipy.spatial import cKDTree, KDTree
import numpy as np
import pandas as pd
import tqdm

from .locs_hdfstore import LocsHDFStore

from matplotlib import pyplot as plt

# Note: this doesn't yet write the linked
#       data back to file

def _near_neighbours(treeA, treeB, radius):
    # indices of all neighbours of treeA in treeB
    indices = treeA.query_ball_tree(treeB, radius)

    # occassionally there will be multiple elements
    # in one tree which will have neighbours in the
    # other or there will be no neighbours. Remove
    # the duplicates and set no neighbours to -1.
    ind = np.ones(len(indices), dtype=np.int64) * -1
    for i, idx in enumerate(indices):
        if len(idx) == 0:
            # no neighbours
            continue
        if len(idx) == 1:
            # one neighbour
            ind[i] = idx[0]
        elif len(idx) > 1:
            # multiple neighbours - the correct one
            # is found by determining the one with the
            # minimum distance
            dist = []
            for el in idx:
                dist.append(treeA.query(treeB.data[el], k=1)[0])
            min_dist, min_dist_idx = min(
                (val, idx) for (idx, val) in enumerate(dist)
            )
            ind[i] = idx[min_dist_idx]
    
    return ind


class Track:
    """
    Representation of a track (a linked set of localisations)
    """
    def __init__(self, loc, first_frame, track_id):
        # DataFrame of localisations
        self.locs = [loc]
        self.x = [loc['x']]
        self.y = [loc['y']]

        self.track_id = track_id
        self.first_added = first_frame
        self.last_added = first_frame
        self.detections = 0
        self._calculate_center()

    def _calculate_center(self):
        mean_x = sum(self.x) / float(len(self.x))
        mean_y = sum(self.y) / float(len(self.y))
        self._center = [mean_x, mean_y]

    def add_localisation(self, loc, frame):
        self.locs.append(loc)
        self.x.append(loc['x'])
        self.y.append(loc['y'])
        self.detections += 1
        self.last_added = frame
        # recalculate the track center
        self._calculate_center()       

    @property
    def center(self):
        return self._center[0:2]


class LinkedList:
    def __init__(self, lhdf):
        self.lhdf = lhdf
        self.centers = []
        self.block_size = 500

    def _write_block(self, block):
        self.lhdf.write_locs(block, key='/linked_table')

    def _track_center(self, track):
        cols = track.columns.values.tolist()
        cols.remove('track_id')
        first_frame = track['frame'].values[0]
        detections = track.shape[0]
        center = pd.DataFrame(track[cols].mean(axis=0)).T
        center['frame'] = first_frame
        center['detections'] = detections
        return center

    def add(self, track):
        self.centers.append(self._track_center(track))
        if len(self.centers) == self.block_size:
            # print(self.block_counter)
            self._write_block(pd.concat(self.centers))
            self.centers = []

    def finish(self):
        self._write_block(pd.concat(self.centers))


def link(locs_path, start, stop, radius=.5,
         max_gap=3, max_length=0, notebook=False):
    """
    Links localisations in successive frames.

    Parameters
    ----------
    locs_path : str
        file path to localisations file
    start : int
        starting frame number
    stop : int
        frame number to stop at
    radius : float
        search radius for near neighbour query in kdtree
    max_gap : int
        maximum number of frames the localisation
        is allowed to disappear and remain part of
        a track
    max_length : int
        maximum allowable length of a track
    notebook : bool
        is this being run from a notebook

    Note
    ----
    Not an exact copy of, but heavily influenced by that
    found here: https://github.com/ZhuangLab/storm-analysis

    Deals with gaps and max length in exactly the same way
    as here: https://github.com/zitmen/thunderstorm
    """
    assert(stop > start)
    pbar_ = tqdm.tqdm
    if notebook:
        pbar_ = tqdm.tqdm_notebook
     
    if radius <=0.0:
        with LocsHDFStore(locs_path) as lhdf:
            lhdf.clear_tracks()
            for locs in lhdf.locs_iterator():
                locs['track_id'] = 0
                lhdf.write_tracks(locs)
    else:
        with LocsHDFStore(locs_path) as lhdf:
            lhdf.clear_links()
            max_ = stop - start
            track_id = 0
            linked = []
            f = start
            print('')
            s = time.time()
            df = lhdf.table
            # for locs in pbar_(
            #     lhdf.locs_frame_iterator(start=start, stop=stop),
            #     total=max_, desc='linking'
            # ):

            for fid, locs in pbar_(
                df.groupby('frame'),
                total=max_, desc='linking'
            ):

                if not locs.empty:
                    deactivate = []
                    for track in linked:
                        if max_length > 0 and (f - track.first_added) >= max_length:
                            deactivate.append(track)
                        if (f - track.last_added) > max_gap:
                            deactivate.append(track)
            
                    for track in deactivate:
                        # write the removed track here
                        # lhdf.write_tracks(track.locs)
                        linked.remove(track)
            
                    centers = np.zeros((len(linked), 2))
                    for tid, track in enumerate(linked):
                        centers[tid, :] = track.center
                    
                    empty_tree = True
                    if centers.any():
                        tracks_tree = cKDTree(centers)
                        locs_tree = cKDTree(locs[['x', 'y']].values)
                        empty_tree = False
                        
                        locs_indices = _near_neighbours(locs_tree, tracks_tree, radius)
                        tracks_indices = _near_neighbours(tracks_tree, locs_tree, radius)

                    for i in range(locs.shape[0]):
                        if not empty_tree:
                            # find the track near neighbour of each loc
                            # and add the loc to the corresponding track
                            # print('i {}'.format(i))
                            if (tracks_indices[locs_indices[i]] == i and
                                locs_indices[i] > -1):
                                track = linked[locs_indices[i]]
                                track.add_localisation(locs.iloc[i], f)
                            elif locs_indices[i] == -1:
                                track = Track(locs.iloc[i], f, track_id)
                                linked.append(track)
                                track_id += 1       
                        else:
                            track = Track(locs.iloc[i], f, track_id)
                            linked.append(track)
                            track_id += 1
                    f += 1
            print(time.time() - s)
        #     # write whatever tracks are left in linked
        #     for track in pbar_(linked, desc='writing links'):
        #         lhdf.write_tracks(track.locs)
        
        # # now write the centers of the tracks as
        # # a new '/linked_table'
        # with LocsHDFStore(locs_path) as lhdf:
        #     num_tracks = track_id
        #     lhdf.key = '/tracks_table'
        #     linked_list = LinkedList(lhdf)
        #     for track in pbar_(
        #         lhdf.track_iterator(num_tracks),
        #         desc='writing tracks'):

        #         linked_list.add(track)

        #     linked_list.finish()

if __name__=='__main__':
    lpath = '.\\test_data\\tetra_speck_beads_time_lapse_0.h5'

    linked = link(lpath, 0, 100, radius=0.5 * 160.0)
    # with LocsHDFStore(lpath) as lhdf:
    #     locs1 = lhdf.get_coords_in_frame(1001)
    #     print(locs1)
    #     print(locs1[138, :])
    #     locs2 = lhdf.get_coords_in_frame(2.0)
    #     print(locs2[137, :])
    #     print(locs2[139, :])
    #     print(locs1[0,:])
    #     tree1 = cKDTree(locs1[:,0:2]/160.0)
    #     tree2 = cKDTree(locs2[:,0:2]/160.0)

    #     indices = tree1.query_ball_tree(tree1, 0.5)
    #     indices1 = tree2.query_ball_tree(tree1, 0.5)
    #     indices2 = tree1.query_ball_tree(tree2, 0.5)

    #     # print('indices1 {}'.format(indices1))
    #     # print('indices2 {}'.format(indices2))

    #     for i in range(locs2.shape[0]):
    #         print('indices[i] {}'.format(indices[i]))
    #         # print('indices1[i] {}'.format(indices1[i]))
    #         # print('indices2[i] {}'.format(indices2[i]))

    #     print(indices1[139])
    # import pims
    # import nd2reader

    # movie = pims.open('.\\test_data\\tubulin647 2d.nd2')
    # print(movie.sizes)
    # # movie.bundle_axes = 'cyx'
    # frame = movie[1000].astype(np.float64)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(frame)
    # plt.plot(locs1[:, 0]/160.0, locs1[:, 1]/160.0, 'rx')
    # # for track in linked:
    #     # plt.plot(track.center[1]/160.0, track.center[0]/160.0, 'bo')
    # # plt.xlim(0, 255)
    # # plt.ylim(255, 0)                
    # plt.show()