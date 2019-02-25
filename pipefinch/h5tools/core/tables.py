import logging
import numpy as np
import h5py

from pipefinch.h5tools.core.h5tools import append_atrributes
# methods to create, append, and i/o into tables (dset objecst of h5py)

logger = logging.getLogger('pipefinch.h5tools.core.tables')


def insert_table(group: h5py.Group, table: np.array, name: str, attr_dict: dict = None) -> h5py.Dataset:
    dset = group.create_dataset(name, data=table)
    if attr_dict is not None:
        append_atrributes(dset, attr_dict)
    return dset


def unlimited_rows_data(group, table_name: str, data: np.array) -> h5py.Dataset:
    """
    Create a table with no max shape, to append data forever
    :param group: h5py Group object. parent group
    :param table_name: str. name of the table
    :param data: np.array with initial data. Can be empty
    :return:
    """
    logger.debug('Creating unbounded table {0} in group {1}'.format(
        group.name, table_name))
    try:
        table = group.create_dataset(table_name,
                                     shape=data.shape,
                                     dtype=data.dtype,
                                     maxshape=(None, None))
        table[:] = data

    except RuntimeError as e:
        if 'Name already exists' in str(e):
            logger.debug(
                'Table {} already exists, appending the data'.format(table_name))
            table = group[table_name]
            append_rows(table, data)
        else:
            raise
    return table


def append_rows(dataset: h5py.Dataset, new_data: np.array):
    '''
    Append rows to an existing table
    :param dataset: h5py Dataset object. Where to append the data
    :param new_data: array. An array to append
    :return:
    '''
    rows = dataset.shape[0]
    more_rows = new_data.shape[0]
    logger.debug('Appending {} rows to dataset {}'.format(
        more_rows, dataset.name))
    dataset.resize(rows + more_rows, axis=0)
    if dataset.size == (rows + more_rows):
        dataset[rows:] = new_data
    else:
        dataset[rows:, :] = new_data


def append_table(data_set: h5py.Dataset, dest: h5py.Dataset, chan_list: np.array = None, chunk_size=8000000):
    # type: (object, object, object, object) -> object
    """
    Append a table onto another table of the same data type (and number of columns)
    :param source_dataset: dataset to merge
    :param dest_dataset: dataset onto which to merge
    :param chan_list:
    :param chunk_size:
    :return:
    """
    samples_data = data_set.shape[0]
    channels_data = data_set.shape[1]
    data_type = data_set.dtype

    logger.debug('Appending {} onto {}'.format(data_set.name, dest.name))
    if chan_list is None:
        logger.debug('Counting channels')
        chan_list = range(channels_data)
    logger.info('Channel count: {}'.format(len(chan_list)))

    samples_chunk = min(chunk_size, samples_data)
    channels_chunk = len(chan_list)

    chunk_buffer = np.empty(
        (samples_chunk, channels_chunk), dtype=np.dtype(data_type))
    chunk_starts = np.arange(0, samples_data, samples_chunk)
    n_chunks = chunk_starts.size

    logger.debug(
        'About to append {} entire chunks plus change'.format(n_chunks - 1))
    for start in chunk_starts:
        logger.debug('Chunk start: {0}'.format(start))
        end = min(start + samples_chunk, samples_data)
        chunk_buffer[0: end - start, :] = load_table_slice(data_set,
                                                           np.arange(
                                                               start, end),
                                                           chan_list)
        append_rows(dest, chunk_buffer[0: end - start, :])

    stored = n_chunks * chunk_buffer.size + \
        chunk_buffer[0: end - start, :].size
    logger.debug('{} elements written'.format(stored))
    return stored


def load_table_slice(table, row_list=None, col_list=None):
    """
    Loads a slice of a h5 dataset.
    It can load sparse columns and rows. To do this, it first grabs the smallest chunks that contains them.
    :param table: dataset of an h5 file.
    :param row_list: list of rows to get (int list)
    :param col_list: list of cols to get (int list)
    :return: np.array of size row_list, col_list with the concatenated rows, cols.
    """
    table_cols = table.shape[1]
    table_rows = table.shape[0]
    d_type = table.dtype

    col_list = np.arange(
        table_cols) if col_list is None else np.array(col_list)
    row_list = np.arange(
        table_rows) if row_list is None else np.array(row_list)

    raw_table_slice = np.empty([np.ptp(row_list) + 1, np.ptp(col_list) + 1],
                               dtype=np.dtype(d_type))
    logger.debug('raw table slice shape {}'.format(raw_table_slice.shape))
    table.read_direct(raw_table_slice,
                      np.s_[np.min(row_list): np.max(row_list) + 1,
                            np.min(col_list): np.max(col_list) + 1])
    # return raw_table_slice
    return_slice = raw_table_slice[row_list -
                                   np.min(row_list), :][:, col_list - np.min(col_list)]
    logger.debug('return table slice shape {}'.format(return_slice.shape))
    return return_slice


# passing stuff to binary
def dset_to_binary_file(data_set, out_file, chan_list=None, chunk_size=8000000, header='bin'):
    """
    :param data_set: a table from an h5 file to write to a binary. has to be daughter of a rec
    :param out_file: binary file - has to be open in 'wb' mode.
    :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    samples_data = data_set.shape[0]
    channels_data = data_set.shape[1]
    data_type = np.dtype(data_set.dtype)
    logger.debug('Ripping dataset from {}'.format(data_set.parent.name))
    if chan_list is None:
        logger.debug('Counting channels')
        chan_list = range(channels_data)
    logger.debug('Channel list: {}'.format(chan_list))

    samples_chunk = min(chunk_size, samples_data)
    channels_chunk = len(chan_list)

    chunk_buffer = np.empty((samples_chunk, channels_chunk), dtype=data_type)
    chunk_starts = np.arange(0, samples_data, samples_chunk)
    n_chunks = chunk_starts.size

    logger.debug('About to store {} entire chunks'.format(n_chunks - 1))
    stored = 0
    for start in chunk_starts:
        logger.debug('Chunk start: {0}'.format(start))
        end = min(start + samples_chunk, samples_data)
        logger.debug(' end: {0}'.format(end))
        chunk_buffer[0: end - start, :] = load_table_slice(data_set,
                                                           np.arange(
                                                               start, end),
                                                           chan_list)
        stored += (chunk_buffer[0: end - start, :]).size
        #logger.info('Chunk buffer dtype {}'.format(chunk_buffer.dtype))
        #logger.info('Chunk dtype dtype {}'.format(data_type))
        if header is 'bin':
            out_file.write(
                chunk_buffer[0: end - start].astype(data_type).tobytes())
        # for the .mda format it needs transposing (their order is nch, n_samples)
        elif header is 'mda':
            buf_write = chunk_buffer[0: end - start].astype(data_type)
            out_file.write(
                buf_write.reshape(buf_write.shape[1], buf_write.shape[0], order='C').tobytes(order='C'))

    # stored = (n_chunks -1) * chunk_buffer.size + \
    #     chunk_buffer[0: end - start, :].size
    logger.debug('{} elements written'.format(stored))
    return stored


def merge_tables(source_file_path, dest_file_path, table_path, chunk_size=8000000):
    logger.debug('Appending table {0} from file {1} into file {2}'.format(table_path,
                                                                          source_file_path,
                                                                          dest_file_path))
    with h5py.File(source_file_path, 'r') as source, h5py.File(dest_file_path, 'r+') as dest:
        append_table(source[table_path], dest[table_path],
                     chunk_size=chunk_size)
        dest.flush()
