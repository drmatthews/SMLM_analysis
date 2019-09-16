from optparse import OptionParser
import os
import csv

import tqdm
import numpy as np
import pandas as pd

from .locs_h5py import LocsHDF


# globals
NSTORM_COLUMNS = ['X', 'Y', 'Z', 'Photons',
                  'Lateral Localization Accuracy',
                  'Frame', 'Length']
THUNDERSTORM_COLUMNS = ['x [nm]', 'y [nm]', 'z [nm]',
                        'intensity [photon]', 'uncertainty [nm]',
                        'frame', 'detections']
HDF_COLS = ['x', 'y', 'z', 'intensity',
            'uncertainty', 'frame',
            'detections']


###
# CSV read, Pandas write - kept in reserve
###
def _write_hdf(h5path, filereader, source, source_cols, chunksize=50000, chunks=1):
    """
    Writes the input localisations to an HDF5 file using Pandas
    HDFStore as chunks (using append). 
    
    Note the data is read in chunks from the source and written
    in chunks to the file to preserve memory.
    """
    # print(source)
    if os.path.exists(h5path):
        os.remove(h5path)

    pbar = tqdm.tqdm
    key = '/linked_table'
    with pd.HDFStore(h5path, mode='w') as store:

        for chunk in pbar(
            _gen_chunks(
                filereader, source_cols, chunksize=chunksize
            ), total=chunks):

            chunk_df = pd.DataFrame(chunk)
            chunk_df.columns = HDF_COLS
            store.append(
                key, chunk_df, format='table',
                data_columns=True, index=False
            )

        metadata = {}
        metadata['n_locs'] = store.get_storer(key).nrows
        metadata['n_frames'] = (
            store.select(key, "columns=['frame']").max()['frame']
        )
        metadata['source'] = source
        metadata['drift_corrected'] = True
        store.get_storer(key).attrs.metadata = metadata


def _to_hdf(fpath, source='nstorm', chunksize=50000):
    if os.path.exists(fpath):

        if 'nstorm' in source and fpath.endswith('txt'):
            delimiter = '\t'
            header = _read_header(fpath, delimiter)
            source_cols = [c for c in NSTORM_COLUMNS
                           if c in header]
        elif fpath.endswith('csv') and 'thunderstorm' in source:
            delimiter = ','
            header = _read_header(fpath, delimiter)

            if 'z' not in header:
                HDF_COLS.remove('z')
                THUNDERSTORM_COLUMNS.remove('z [nm]')
            if 'detections' not in header:
                HDF_COLS.remove('detections')
                THUNDERSTORM_COLUMNS.remove('detections')

            source_cols = [c for c in THUNDERSTORM_COLUMNS if c in header]

        num_lines = _linecount(fpath)
        chunks = num_lines // chunksize
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fname = os.path.basename(fpath)
            basename, _ = os.path.splitext(fname)
            h5path = os.path.join(os.path.dirname(fpath), basename + '.h5')

            write_hdf_h5py(h5path, reader, source, source_cols,
                       chunksize=chunksize, chunks=chunks)
###

###
# CSV read helpers - kept in reserve in case
# reading changes to csv and writing to h5py
###
def _linecount(filename):
    """
    Use unbuffered interface to read line number.
    Useful for updating progress bar.
    """
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def _gen_chunks(reader, source_cols, chunksize=100):
    """ 
    Chunk generator. Take a CSV `DictReader` and yield
    `chunksize` sized slices. 
    """
    chunk = []
    for index, line in enumerate(reader):
        if (index % chunksize == 0 and index > 0):
            yield chunk
            del chunk[:]
        try:
            row = tuple(float(line[key]) for key in source_cols)
            chunk.append(row)
        except:
            continue
    yield chunk

def _gen_frames(reader, source_cols, frame_col):
    """ 
    Frame generator. Take a CSV `DictReader` and yield
    slices which represent a single frame in a SMLM movie. 
    """

    frame_no = next(reader)[frame_col]
    frame = []
    for index, line in enumerate(reader):
        if (line[frame_col] != frame_no):
            yield frame
            del frame[:]
            frame_no = line[frame_col]
        try:
            row = tuple(float(line[key]) for key in source_cols)
            frame.append(row)
        except:
            continue
    yield frame


def _read_header(fpath, delimiter):
    """
    Read the file header - useful for
    getting column names.
    """
    with open(fpath, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
    return header
### end helpers

###
# keep this in reserve if locs interface changes to use h5py instead of pandas
def write_hdf_h5py(h5path, filereader, source, source_cols, chunksize=50000, chunks=1):
    """
    Writes the input localisations to an HDF5 file using h5py
    to write data one SMLM movie frame at a time (slow). 
    
    Note the data is read in chunks from the source and written
    in chunks to the file to preserve memory.
    """
    # print(source)
    if os.path.exists(h5path):
        os.remove(h5path)

    frame_col = [col for col in source_cols if 'frame' in col.lower()][0]
    last_frame = filereader[-1][frame_col]
    print(last_frame)
    pbar = tqdm.tqdm
    with LocsHDF(h5path, mode='write') as lhdf:
        dtype = [(col, 'f8') for col in HDF_COLS]
        for frame in pbar(
            _gen_frames(filereader, source_cols),
            total=20000):
            locs = np.array(frame, dtype=dtype)
            lhdf.write_locs(locs)
###

def write_hdf(h5path, filereader, source, source_cols, hdf_cols):
    """
    Writes the input localisations to an HDF5 file using Pandas
    HDFStore as chunks (using append). 
    
    Note the data is read in chunks from the source and written
    in chunks to the file to conserve memory.

    :param h5path:      the path to write to
    :type  h5path:      str
    :param filereader:  filereader iterator object
    :type  filereader:  pandas FileReader
    :param source:      the software that generated the original file
    :type  source:      str
    :param source_cols: columns to use in the source file
    :type  source_cols: list
    :param hdf_cols:    column names in the HDF5 data
    :type  hdf_cols:    list
    """
    # print(source)
    if os.path.exists(h5path):
        os.remove(h5path)

    # if linking/merging has not 
    # already been done, give the user
    # the choice of running linking on the
    # table
    if 'detections' in hdf_cols:
        key = '/linked_table'
    else:
        key = '/table'

    with pd.HDFStore(h5path, mode='w') as store:

        for c, chunk in enumerate(filereader):
            chunk = chunk[source_cols]
            chunk.columns = hdf_cols
            store.append(
                key, chunk, format='table',
                data_columns=True, index=False
            )

        metadata = {}
        metadata['n_locs'] = store.get_storer(key).nrows
        metadata['n_frames'] = (
            store.select(key, "columns=['frame']").max()['frame']
        )
        metadata['source'] = source
        metadata['drift_corrected'] = True
        store.get_storer(key).attrs.metadata = metadata


def to_hdf(fpath, source='nstorm'):
    """
    Convert a input table of localisation microscopy data to an
    HDF format file - the data columns are reduced to those in
    HDF_COLUMNS. This can accept data from either NSTORM or
    ThunderSTORM.

    :param fpath:  full path to the source file being converted
    :type  fpath:  str
    :param source: the software that generated the original file
    :type  source: str
    """
    if os.path.exists(fpath):

        hdf_cols = ['x', 'y', 'z', 'intensity',
                    'uncertainty', 'frame', 'detections']

        if 'nstorm' in source and fpath.endswith('txt'):
            delimiter = '\t'
            header = _read_header(fpath, delimiter)
            source_cols = [c for c in NSTORM_COLUMNS
                            if c in header]
            read_fnc = pd.read_table

        elif fpath.endswith('csv') and 'thunderstorm' in source:
            delimiter = ','
            header = _read_header(fpath, delimiter)

            if 'z' not in header:
                hdf_cols.remove('z')
                THUNDERSTORM_COLUMNS.remove('z [nm]')
            if 'detections' not in header:
                hdf_cols.remove('detections')
                THUNDERSTORM_COLUMNS.remove('detections')

            source_cols = [c for c in THUNDERSTORM_COLUMNS if c in header]            
            read_fnc = pd.read_csv

        kwargs = dict(
            delimiter=delimiter, header=0,
            usecols=source_cols, chunksize=50000
        )
        reader = read_fnc(fpath, **kwargs)

        fname = os.path.basename(fpath)
        basename, _ = os.path.splitext(fname)
        h5path = os.path.join(os.path.dirname(fpath), basename + '.h5')

        write_hdf(h5path, reader, source, source_cols, hdf_cols)
    else:
        raise OSError('File not found')


def batch_convert_to_hdf(folder, source='nstorm', read_chunks=False):
    """
    Convert a whole folder of localisation files to HDF5 format
    """
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith("csv") or filename.endswith("txt"):
                print(
                    ("converting {0} file {1} to HDF5".
                    format(source, filename))
                )
                fpath = os.path.join(folder, filename)
                to_hdf(fpath, source=source)


if __name__ == '__main__':
    
    import time
    parser = OptionParser(
        usage='Usage: %prog [options] <localisations>'
    )
    parser.add_option(
        '-d', '--directory',
        metavar='DIRECTORY', dest='folder',
        help='directory of files to batch convert'
    )
    parser.add_option(
        '-s', '--source', metavar='SOURCE', dest='source',
        type='str', default='nstorm',
        help=('in which software were the localisations'
              'created (nstorm or thunderstorm)')
    )

    (opts, args) = parser.parse_args()
    try:
        fpath = args[0]
        start = time.time()
        to_hdf(fpath, source=opts.source)
        print(time.time() - start)
    except IndexError:
        folder = opts.folder
        batch_convert_to_hdf(folder, source=opts.source)
    except IndexError:
        parser.error('pass a filename or folder name')