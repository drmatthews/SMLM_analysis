import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from read_roi import read_roi_file
from read_roi import read_roi_zip

from .utils import (kdtree_nn,
                    polygon_area,
                    polygon_perimeter)

import warnings
warnings.filterwarnings("ignore")


def collect_filenames(folder, dataset_name):
    fnames = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            fnames.append(filename)
    return fnames


def write_stats(output_path, data, dataset_name):
    if isinstance(data, dict):
        df = pd.DataFrame()
        for key, value in data.items():
            df[key] = value
    else:
        df = data

    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    if os.path.exists(output_path):
        book = load_workbook(output_path)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name=dataset_name, index=False)
    writer.save()


def collate_dataset_stats(out_dir, dataset_name, dataset_type, skip=[None]):
    print(dataset_name)
    df_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx') and (filename not in skip):
            if ((dataset_type in filename) and
                    (dataset_name in filename.lower())):

                df_list.append(
                    pd.read_excel(
                        os.path.join(out_dir, filename),
                        sheet_name='cluster statistics'
                    )
                )

    return pd.concat(df_list)


def cluster_near_neighbours(out_dir, dataset_name, dataset_type):
    print(dataset_name)
    nn_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx'):
            if (dataset_type in filename) and (dataset_name in filename):
                print(filename)
                df = pd.read_excel(
                    os.path.join(out_dir, filename),
                    sheet_name='cluster coordinates')

                nn = kdtree_nn(
                    df.as_matrix(columns=['x [nm]', 'y [nm]']))
                nn_list.append(nn)

    nn_df = pd.DataFrame(np.concatenate(nn_list, axis=0))
    nn_df.columns = ['Distance [nm]']

    return nn_df


def percentage_of_total(folder,
                        sheet_name,
                        dataset_name,
                        condition='eq',
                        cluster_column='lk'):
    percent = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(os.path.join(folder, filename),
                               sheet_name=sheet_name)
            if 'eq' in condition:
                filtered = float(df[df[cluster_column] == -1].shape[0])
            elif 'gt' in condition:
                filtered = float(df[df[cluster_column] > -1].shape[0])
            print("number of filtered locs: {0}".format(filtered))
            total = float(df.shape[0])
            print("total number of localisations: {0}".format(total))
            p = filtered / total
            print("percentage: {0}".format(p))
            percent.append(p)
    return np.array(percent)


def percentage_objects_with_clusters(folder, sheet_name, dataset_name):
    percent = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(os.path.join(folder, filename),
                               sheet_name=sheet_name)

            objects = float(df[df['object_id'] > -1].shape[0])
            clusters = float(df[df['cluster_id'] > -1].shape[0])

            print("number of object locs: {0}".format(objects))
            print("number of cluster locs: {0}".format(clusters))
            p = objects / clusters
            print("percentage: {0}".format(p))
            percent.append(p)
    return np.array(percent)


def mean_per_image(folder, dataset_name, parameter, sheet_name):
    mean = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(
                os.path.join(folder, filename),
                sheet_name=sheet_name)
            if 'labels' in df:
                df = df[df['labels'] > -1]
            mean.append(df[parameter].mean())
    return np.array(mean)


def mean_near_neighbour_distance(folder, dataset_name, sheet_name):
    mean = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(
                os.path.join(folder, filename),
                sheet_name=sheet_name)
            cluster_mean = []
            for cid, c in df.groupby('cluster_id'):
                cluster_mean.append(c['nn distance [nm]'].mean())
            mean.append(sum(cluster_mean) / len(cluster_mean))
    return np.array(mean)


def collect_stats(df, col):
    total = float(df.shape[0])
    not_noise = df[df[col] != -1]
    percentage = float(not_noise.shape[0]) / total
    objects_with_clusters = (
        float(df[(df['object_id'] != -1) & (df['cluster_id'] != -1)].shape[0])
    )
    labels = not_noise[col].unique()
    area = []
    perimeter = []
    occupancy = []
    stats = {}
    stats['label'] = labels
    for m in labels:
        group = df[df[col] == m]
        points = group.as_matrix(['x [nm]', 'y [nm]'])
        # area.append(group['area'].sum())
        area.append(polygon_area(points))
        perimeter.append(polygon_perimeter(points))
        occupancy.append(len(group.index))

    stats['area'] = area
    stats['perimeter'] = perimeter
    stats['occupancy'] = occupancy
    stats['percentage {0}s'.format(col[:len(col) - 3])] = percentage
    stats['not_noise'] = float(not_noise.shape[0])
    stats['objects_with_clusters'] = objects_with_clusters
    return stats


def import_cluster_stats(path, sheetname=None):
    if path.endswith('xlsx'):
        sn = 'image cluster stats'
        if sheetname:
            sn = '{0} cluster stats'.format(sheetname)
        try:
            df = pd.read_excel(path, sheet_name=sn)
            return df
        except ValueError:
            sn = 'image cluster stats'
            df = pd.read_excel(path, sheet_name=sn)
            return df
        except ValueError:
            return None
    else:
        raise ValueError("Input data must be in Excel format")


def batch_stats(folder, conditions, use_roi=False):

    for filename in os.listdir(folder):
        if filename.endswith('xlsx'):
            stats_path = os.path.join(folder, filename)
            basename = os.path.splitext(filename)[0]

            if use_roi:
                roi_zip_path = os.path.join(folder, basename +
                                            '_roiset.zip')
                roi_file_path = os.path.join(folder,
                                             basename + '_roiset.roi')
                if os.path.exists(roi_zip_path):
                    rois = read_roi_zip(roi_zip_path)
                elif os.path.exists(roi_file_path):
                    rois = read_roi_file(roi_file_path)
                else:
                    raise ValueError(("No ImageJ roi file exists -"
                                      "you should put the file in the same"
                                      "directory as the data"))

                stats = []
                for roi in rois.keys():
                    stats.append(import_cluster_stats(stats_path),
                                 sheetname=roi)
                stats_df = pd.concat(stats)
            else:
                stats_df = import_cluster_stats(stats_path)

            fnames = [f for f in listdir(folder) if isfile(join(folder, f))]
            outpath = os.path.join(folder, 'cluster_statistics_test.xlsx')
            for condition in conditions:
                condition_fnames = [fname for fname in fnames if condition in fname]
                cluster_stats = []
                for cf in condition_fnames:
                    cluster_stats.append(stats[cf])

                cddf = pd.DataFrame(condition_fnames)
                cddf.columns = ['filename']
                csdf = pd.concat(cluster_stats, axis=0)
                data = pd.concat([cddf, csdf.reset_index(drop=True)], axis=1)
                statistics.write_stats(stats_path, data, condition)
