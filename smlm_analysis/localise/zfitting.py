import os
import yaml
from datetime import date

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
import scipy
import scipy.optimize
import tqdm

from ..utils.locs_hdfstore import LocsHDFStore


def _interpolate_nan(df, indices):

    for i in indices:
        df.loc[i, :] = np.nan
    df = df.sort_index()
    return df.interpolate(
        method='linear', limit_direction='both', limit=2
    )



# Least Squares fitting
def fit_function_LS(data, params, z, fn):
    """
    Wrapper around scipy.optimize.leastsq for fitting
    z defocus function to calibration data.
    """
    result = params
    errorfunction = lambda p: fn(*p)(z) - data
    good = True
    [result, cov_x, infodict, mesg, success] = (
        scipy.optimize.leastsq(
            errorfunction, params, full_output = 1, maxfev = 500
        )
    )
    err = errorfunction(result)
    err = scipy.sum(err * err)
    if (success < 1) or (success > 4):
        print( "Fitting problem!", success, mesg)
        good = False
    return [result, cov_x, infodict, good]


def z_defocus(w0, c, d, A, B):
    """
    z defocus function used for both bead calibration data
    and for finding the best z position for a given PSF
    width.
    """
    return lambda z: (
        w0 * np.sqrt(1 + \
        ((z - c)/d)**2 + A*((z - c)/d)**3 + B*((z - c)/d)**4)
    )


def fit_z_defocus(data, z, w0, c, d, A, B):
    """
    Does least squares fitting of z defocus.
    """
    params = [w0, c, d, A, B]
    return fit_function_LS(data, params, z, z_defocus) 


def fourth_poly(a, b, c, d, e):
    """
    Fourth order polynomial function used for finding
    best z position for a given PSF width.
    """
    return lambda z: a*z**4 + b*z**3 + c*z**2 + d*z + e


def z_dist(z, wx_cal, wy_cal, wx, wy):
    """
    Function to be minimised when finding best z position
    for a given PSF width.
    """
    return (sqrt(wx) - sqrt(wx_cal(z)))** 2 + \
           (sqrt(wy) - sqrt(wy_cal(z)))** 2   


def minimise_distance(wx_wy, z_range, calibration, fit_type):
    """
    Wrapper around scipy.optimize.minimize_scalar for
    minimising `z_dist`.
    """
    if fit_type == 'fourth_poly':
        func_x = fourth_poly(*calibration['x'])
        func_y = fourth_poly(*calibration['y'])
    elif fit_type == 'zdefocus':
        func_x = z_defocus(*calibration['x'])
        func_y = z_defocus(*calibration['y'])

    zbounds = (min(z_range), max(z_range))
    result = scipy.optimize.minimize_scalar(
        z_dist, args=(func_x, func_y, wx_wy[0], wx_wy[1]),
        method='bounded',
        bounds=zbounds
    )
    return result.x


def find_best_z(locs, z_range, zcal, fit_type):
    z = np.zeros(locs.shape[0])
    i = 0
    for _, row in locs.iterrows():
        z[i] = minimise_distance(
            row[['wx', 'wy']].values,
            z_range, zcal, fit_type
        )
        i += 1
    return z


# nicked straight outta picasso
def plot_calibration(locs, frames, calibration, z_range, zcal_path):

    # get z and PSF width values

    z = locs['z'].values
    wx = locs['wx'].values
    wy = locs['wy'].values
    mean_wx_wy = (
        locs[['wx', 'wy', 'frame']].groupby('frame').mean()
    )
    fillnan = list(set(frames) - set(mean_wx_wy.index.tolist()))
    mean_wx_wy = _interpolate_nan(mean_wx_wy, fillnan)
    mean_wx = mean_wx_wy['wx'].values
    mean_wy = mean_wx_wy['wy'].values

    # plot
    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.plot(z_range, mean_wx, ".-", label="x")
    plt.plot(z_range, mean_wy, ".-", label="y")
    plt.plot(
        z_range, np.polyval(calibration['x'], z_range),
        "0.3", lw=1.5, label="x fit"
    )
    plt.plot(
        z_range, np.polyval(calibration['y'], z_range),
        "0.3", lw=1.5, label="y fit"
    )
    plt.xlabel("Stage position")
    plt.ylabel("Mean spot width/height")
    plt.xlim(z_range.min(), z_range.max())
    plt.legend(loc="best")

    ax = plt.subplot(222)
    plt.scatter(wx, wy, c="k", lw=0, alpha=0.1)
    plt.plot(
        np.polyval(calibration['x'], z_range),
        np.polyval(calibration['y'], z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    plt.plot()
    ax.set_aspect("equal")
    plt.xlabel("Spot width")
    plt.ylabel("Spot height")
    plt.legend(loc="best")

    plt.subplot(223)
    plt.plot(z, wx, ".", label="x", alpha=0.2)
    plt.plot(z, wy, ".", label="y", alpha=0.2)
    plt.plot(
        z_range, np.polyval(calibration['x'], z_range),
            "0.3", lw=1.5, label="calibration"
    )
    plt.plot(
        z_range, np.polyval(calibration['y'], z_range),
        "0.3", lw=1.5
    )
    plt.xlim(z_range.min(), z_range.max())
    plt.xlabel("Estimated z")
    plt.ylabel("Spot width/height")
    plt.legend(loc="best")

    ax = plt.subplot(224)
    first_frame = locs['frame'].iloc[0]
    for gid, group in locs.groupby('frame'):
        idx = group['frame'].values - first_frame
        plt.plot(
            z_range[idx.astype(np.int64)],
            group['z'].values,
            ".k", alpha=0.1
        )
    plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    plt.xlim(z_range.min(), z_range.max())
    plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    plt.xlabel("Stage position")
    plt.ylabel("Estimated z")
    plt.legend(loc="best")

    plt.tight_layout(pad=2)

    dirname = zcal_path[:-5]
    plt.savefig(dirname + ".png", format='png', dpi=300)
    plt.show()



def calibrate(locs_path, zcal_path, z_step, fit_type, notebook, guess=None):
    """
    Determine z look-up-table for bead calibration recorded
    using astigmatism method.
    """
    print('Calibrating z...')
    with LocsHDFStore(locs_path) as lhdf:
        # if no widths are in the file, then
        # fitting hasn't been done

        # calculate z range
        tot_z = (lhdf.n_frames - 1) * z_step
        frame_range = np.arange(lhdf.n_frames)
        z_range = -(
            frame_range * z_step - tot_z / 2
        )

        locs = lhdf.get_localisations()
        z = locs['z'].values
        wx = locs['wx'].values
        wy = locs['wy'].values
        start = int(lhdf.frames[0])
        stop = int(lhdf.frames[-1])
        frames = list(range(start, stop + 1))
        
        # this is for the outlier removal
        filtered = []
        for gid, group in locs.groupby('frame'):

            mean = group[['wx', 'wy']].mean()
            var = group[['wx', 'wy']].var()

            keep = (
                ((group['wx'] - mean['wx'])**2 < var['wx']) &
                ((group['wy'] - mean['wy'])**2 < var['wy'])
            )
            filtered.append(group[keep])

        locs = pd.concat(filtered, ignore_index=True)
        mean_wx_wy = (
            locs[['wx', 'wy', 'frame']].groupby('frame').mean()
        )
        fillnan = list(set(frames) - set(mean_wx_wy.index.tolist()))
        mean_wx_wy = _interpolate_nan(mean_wx_wy, fillnan)
        wx = mean_wx_wy['wx'].values
        wy = mean_wx_wy['wy'].values
        if fit_type == 'fourth_poly':
            calib_x = np.polyfit(z_range, wx, 4)
            calib_y = np.polyfit(z_range, wy, 4)
        elif fit_type == 'zdefocus':
            if guess:
                calib_x, cov_x, infodict, good = (
                    fit_z_defocus(wx, z_range, *guess)
                )
                calib_y, cov_y, infodict, good = (
                    fit_z_defocus(wy, z_range, *guess)
                )
            else:
                raise ValueError('Provide an inital guess for z defocus')

    calibration = {
        'date': date.today(),
        'description': "fourth order polynomial fit",
        'x': [float(x) for x in calib_x],
        'y': [float(y) for y in calib_y]
    }
    # write the calibration to file
    with open(zcal_path, 'w') as calfile:
        yaml.dump(calibration, calfile, default_flow_style=False)

    # determine z positions from calibration
    print("")
    print("Fitting z positions...")
    z = find_best_z(locs, z_range, calibration, fit_type)

    # plots to assess quality of calibration
    print("")
    print("Creating plots...")
    plot_calibration(locs, frames, calibration, z_range, zcal_path)


def fitz(locs_path, zcal_path, z_step, fit_type, notebook):
    """
    Find the z positions for a table of localisations recorded
    using the astigmatism method.
    """
    if os.path.exists(zcal_path):
        pbar = tqdm.tqdm
        if notebook:
            pbar = tqdm.tqdm_notebook

        with open(zcal_path, 'r') as zcal_file:
            zcal = yaml.load(zcal_file, Loader=yaml.FullLoader)

        with LocsHDFStore(locs_path) as lhdf:
            # if no widths are in the file, then
            # fitting hasn't been done

            tot_z = (lhdf.n_frames - 1) * z_step
            frame_range = np.arange(lhdf.n_frames)
            z_range = -(
                frame_range * z_step - tot_z / 2
            )
            current_key = lhdf.key
            metadata = lhdf.metadata

            for locs in pbar(
                lhdf.locs_frame_iterator(),
                total=lhdf.n_frames,
                desc='Determining z positions'):

                wx_wy = locs[['wx', 'wy']]
                locs['z'] = find_best_z(wx_wy, z_range, zcal, fit_type)
                lhdf.write_locs(locs, key='/temp_table')

            # remove the old table and rename the new one
            # can't find a better way to rewrite a column
            # since all the data is in a single table
            lhdf.remove_table(current_key)
            lhdf.rename_table('/temp_table', current_key[1:])
            lhdf.write_metadata(metadata, key=current_key)
    else:
        raise ValueError('Z calibration not found')


if __name__=='__main__':
    locs_path = '.\\test_data\\z_7448_647_Calib.h5'
    zcal_path = '.\\test_data\\z_7448_647_Calib_zcal.yaml'
    calibrate(locs_path, zcal_path, 5.0, 'fourth_poly', False)
