import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils.locs_hdfstore import LocsHDFStore
from ..render.rendering import hist_2d, hist_3d, image_2d, gaussian_2d


def run_3d(locs_path, z_step=100.0, scale=5, camera_pixel=160, camera_format=(256,256)):

    bins = [f * scale for f in camera_format] # num pixels in reconstructed image
    xy_scale = float(camera_pixel / scale)

    with LocsHDFStore(locs_path) as lhdf:
        df = lhdf.table

        x = df['x'].values
        y = df['y'].values
        z = df['z'].values

        z_min = df['z'].min()
        print(z_min)
        z_max = df['z'].max()
        print(z_max)
        z_bins = int((z_max - z_min) / z_step)
        print(z_bins)
        bins.append(z_bins)

        i_x = np.floor(x / xy_scale).astype(np.int32)
        i_y = np.floor(y / xy_scale).astype(np.int32)
        i_z = np.floor((z - z_min) / z_step).astype(np.int32)
        n = x.shape[0]
        print(n)
        im = np.zeros(bins, dtype=np.int32)
        start = time.time()
        hist_3d(i_x, i_y, i_z, n, im)
        print(time.time() - start)

    print(im.shape)

    plt.figure()
    plt.imshow(np.max(im, axis=0))
    plt.show()

def run_2d(locs_path, scale=2, camera_pixel=160, camera_format=(256,256)):

    bins = [int(f * scale) for f in camera_format] # num pixels in reconstructed image
    bin_size = float(camera_pixel / scale)
    print(bin_size)
    with LocsHDFStore(locs_path) as lhdf:
        # df = lhdf.table
        df = lhdf.get_in_frame_range(start=0, stop=20000)

        x = df['x'].values
        y = df['y'].values
        i_x = np.floor(x / bin_size).astype(np.int32)
        i_y = np.floor(y / bin_size).astype(np.int32)
        n = x.shape[0]
        print(n)
        im = np.zeros(bins, dtype=np.int32)
        # start = time.time()
        # hist_2d(i_x, i_y, n, im)
        # print(time.time() - start)
        start = time.time()
        # im = image_2d(x, y, bins[0], bins[1], bin_size)
        hist_2d(i_x, i_y, n, im)
        print(time.time() - start)

    plt.figure()
    # plt.imshow(im)
    # plt.plot(i_x, i_y, 'rx')
    plt.imshow(im, vmax=10.0)
    plt.show()

def run_gaussian2d(locs_path, scale=2, camera_pixel=160, camera_format=(256,256)):

    bins = [int(f * scale) for f in camera_format] # num pixels in reconstructed image
    bin_size = float(camera_pixel / scale)

    with LocsHDFStore(locs_path) as lhdf:
        df = lhdf.table
        # df = lhdf.get_in_frame_range(start=10000, stop=10999)

        x = df['x'].values
        y = df['y'].values
        px = df['uncertainty'].values
        py = df['uncertainty'].values

        start = time.time()
        im = gaussian_2d(x, y, px, py, bins[0], bins[1], bin_size, min_blur_width=0.0001)
        print(time.time() - start)

    plt.figure()
    # plt.imshow(im)
    # plt.plot(i_x, i_y, 'rx')
    plt.imshow(im, vmax=50)
    plt.show()

if __name__=='__main__':

    # locs_path = '.\\test_data\\ts_3D.h5'
    # # run_2d(locs_path)
    # run_3d(locs_path, camera_pixel=100, camera_format=(64,64))

    locs_path = '.\\test_data\\tubulin647 2d.h5'
    # run_gaussian2d(locs_path)
    run_2d(locs_path)