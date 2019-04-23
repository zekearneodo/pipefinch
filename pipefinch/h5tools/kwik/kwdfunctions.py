import logging
import os
import h5py
import numpy as np
import pandas as pd
import tempfile
import shutil
import contextlib


from numpy.lib import recfunctions as rf
from numba import jit

from pipefinch.neural.convert.mdautil import update_mda_hdr, write_mda_hdr_explicit, mda_fun_dict
from pipefinch.h5tools.core.h5tools import h5_decorator, list_subgroups, obj_attrs_2_dict_translator
from pipefinch.h5tools.core.tables import dset_to_binary_file
from pipefinch.h5tools.kwik.kutil import get_rec_list, parse_tstamp

from intan2kwik.core.file import util as fu
from intan2kwik.kwd import which_board, intan_to_kwd_multirec
from tqdm import tqdm_notebook as tqdm

logger = logging.getLogger('pipefinch.h5tools.kwik.kwdfunctions')


@jit
def get_slice_array(dset: np.ndarray, starts: np.ndarray, span: np.int, chan_list) -> np.ndarray:
    n_slices = starts.size
    #n_chan = dset.shape[1]
    n_chan = chan_list.size
    slices_array = np.zeros([n_slices, span, n_chan])
    for i, start in enumerate(starts):
        slices_array[i, :, :] = dset[start: start+span, chan_list]
    return slices_array

@h5_decorator(default_mode='r')
def collect_frames_fast(kwd_file, recs_list, starts, span, chan_list):
    recs = np.unique(recs_list)
    all_frames_list = []
    for i_rec, rec in tqdm(enumerate(recs), total=recs.size):
        starts_from_rec = starts[recs_list == rec]
        dset = get_data_set(kwd_file, rec)
        n_samples = dset.shape[0]
        valid_starts = starts_from_rec[(starts_from_rec > 0)
                                       & (starts_from_rec + span < n_samples)]
        if valid_starts.size < starts_from_rec.size:
            logger.warn('Some frames were out of bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(
                valid_starts.size))

        # get the dataset slices for only the channel list
        this_rec_frames = get_slice_array(dset, valid_starts, span, chan_list)
        all_frames_list.append(this_rec_frames)

    try:
        all_frames_array = np.concatenate(all_frames_list, axis=0)
    except ValueError:
        raise
        # logger.warn('Failed to collect stream frames, return is nan array')
        # zero_dset_shape = get_data_set(kwd_file, rec).shape
        # all_frames_array = np.empty([1, *zero_dset_shape])
        # all_frames_array[:] = np.nan
    return all_frames_array


def get_data_set(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: h5 dataset object with
    """
    #logger.debug('Getting dataset from rec {}'.format(rec))
    return kwd_file['/recordings/{}/data'.format(rec)]



def get_data_chunk(kwd_file: h5py.File, rec: np.int, start: np.int, span: np.int,
                   chan_list: np.array, table_name: str='data') -> np.array:
    """get a chunk of a dataset from a particular rec in a kwd/kwik file.

    Arguments:
        kwd_file {h5py.File} -- h5py.File object (open in 'r' mode at least)
        rec {np.int} -- id of recording group
        start {np.int} -- start (samples) within the rec group
        span {np.int} -- number of samples to read from start positions
        chan_list {np.array} -- list of channel indices from the table to read

    Keyword Arguments:
        table_name {str} -- name of the table (e.g. 'data', 'dig_in', ...) (default: {'data'})

    Returns:
        np.array -- array with the chunk of data (n_samples x n_col)
    """
    dset = kwd_file['/recordings/{}/{}'.format(rec, table_name)]
    place_holder = np.zeros([span, chan_list.size])
    place_holder[:] = np.nan

    block_read = dset[start: start + span, chan_list]
    #logger.info('read block shape {}'.format(block_read.shape))

    if start < 0:
        shift = start * (-1)
        start = 0
        span = span - shift
        logger.info('negative start, there will be nans')
    else:
        shift = 0

    span = block_read.shape[0]
    #logger.info('span {}'.format(place_holder[shift: span, :].shape))
    place_holder[shift: span, :] = block_read
    return place_holder


def get_digital_chunk(kwd_file: h5py.File, rec: np.int, start: np.int, span: np.int,
                      chan_list: np.array) -> np.array:
    """get a chunk of a dataset from a particular rec in a kwd/kwik file.

    Arguments:
        kwd_file {h5py.File} -- h5py.File object (open in 'r' mode at least)
        rec {np.int} -- id of recording group
        start {np.int} -- start (samples) within the rec group
        span {np.int} -- number of samples to read from start positions
        chan_list {np.array} -- list of channel indices from the table to read

    Returns:
        np.array -- array with the chunk of data (n_samples x n_col)
    """
    return get_data_chunk(kwd_file, rec, start, span, chan_list, table_name='dig_in')


@h5_decorator(default_mode='r')
def get_rec_attrs(kwd_file, rec) -> dict:
    rec_attrs = dict()
    # names of metadata groups and locations in the rec group:
    fields_locs = {'rec_group': '/recordings/{}'.format(rec),
                   'app_data': '/recordings/{}/application_data'.format(rec),
                   'cont_data': '/recordings/{}/data'.format(rec)}

    for name, location in fields_locs.items():
        rec_attrs[name] = obj_attrs_2_dict_translator(kwd_file[location])
    return rec_attrs


@h5_decorator(default_mode='r')
def get_data_size(kwd_file, rec):
    return get_data_set(kwd_file, rec).shape[0]


@h5_decorator(default_mode='r')
def get_rec_sizes(kwd_file):
    rec_list = get_rec_list(kwd_file)
    rec_sizes = {rec: get_data_size(kwd_file, rec)
                 for rec in rec_list}
    return rec_sizes


@h5_decorator(default_mode='r')
def get_rec_starts(kwd_file):
    logger.debug('Getting rec_starts')
    rec_sizes = get_rec_sizes(kwd_file)
    #logger.debug('rec sizes {}'.format(rec_sizes))
    starts_vec = np.array(list(rec_sizes.values())).cumsum()
    #logger.debug('starts vector {}'.format(starts_vec))
    starts_vec = np.hstack([0, starts_vec[:-1]])
    rec_starts = {rec: r_start for r_start,
                  rec in zip(starts_vec, rec_sizes.keys())}
    return rec_starts

@h5_decorator(default_mode='r')
def get_frames(kwd_file, starts: np.ndarray, recs: np.ndarray, span: int, chan_names_list: np.ndarray, 
    sess_meta_pd: pd.DataFrame):
    
    all_rec = np.unique(recs)

    all_rec_frames = []
    for r in all_rec:
        chan_list = rec_chan_idx(sess_meta_pd, r, chan_names_list, block='analog')
        starts_rec = starts[recs == r]
        starts_rec.sort()
        frames = list(map(lambda x: get_data_chunk(kwd_file, r, x, span, chan_list), 
        starts_rec))
        all_rec_frames.append(np.stack(frames, axis=0))

    all_frames_array = np.concatenate(all_rec_frames, axis=0)
    return all_frames_array

@h5_decorator(default_mode='r')
def get_data_type(kwd_file: h5py.File):
    all_rec_list = get_rec_list(kwd_file)
    first_dataset = get_data_set(kwd_file, all_rec_list[0])
    return first_dataset.dtype

@h5_decorator(default_mode='r')
def get_all_rec_meta(kwd_file: h5py.File) -> pd.DataFrame:
    """[summary]
    Arguments:
        kwd_file {h5py.File} -- h5py file open in 'r' mode (kwd file)

    Returns:
        pd.DataFrame -- pandas dataframe with metadata across all recs.
    """
    # list all the recs in the file
    all_rec_list = get_rec_list(kwd_file)
    all_rec_meta_list = list(
        map(lambda rec: get_rec_attrs(kwd_file, rec), all_rec_list))

    # example structure of the meta dictionary
    example_dict = all_rec_meta_list[0]
    all_meta_dict = dict()
    # make a flattened dict
    for k, sub_dict in example_dict.items():
        for k_s in sub_dict.keys():
            all_meta_dict[k_s] = [m[k][k_s] for m in all_rec_meta_list]

    all_meta_pd = pd.DataFrame(all_meta_dict)
    # all_meta_pd.head()
    all_meta_pd['samples_count'] = list(get_rec_sizes(kwd_file).values())
    # todo: refine the dict
    # timestaps from string to timestamp
    all_meta_pd['start_time'] = all_meta_pd['start_time'].apply(parse_tstamp)
    return all_meta_pd


def get_rec_range(pd_meta: pd.DataFrame, period_start: str, period_end: str) -> np.array:
    if not pd_meta.index.name == 'start_time':
        pd_meta.set_index('start_time', inplace=True)
    rec_names_arr = pd_meta.between_time(
        period_start, period_end)['name'].values
    pd_meta.reset_index(drop=False)
    return rec_names_arr


def get_sampling_rate(meta_pd: pd.DataFrame, rec_name) -> np.float:
    return meta_pd.loc[meta_pd.name == rec_name, 'sample_rate'].values[0]


def get_all_chan_names(meta_pd: pd.DataFrame, chan_filt: np.ndarray = np.empty(0),
                       block: str ='analog') -> np.ndarray:
    """[summary]

    Arguments:
        meta_pd {pd.DataFrame} -- pandas dataframe with the metadata through the session

    Keyword Arguments:
        chan_filt {np.ndarray} -- list of search-strings to filter channels,
        can be used to find particular channels, or channel groups 
        (e.g np.array(['A-', ADC'] will pick ADC channels and ephys port A) 
        (default: {np.empty(0)})
        block {str} -- channel block ('analog', 'digital', 'aux') (default: {'analog'})
        (note, 'analog' is for ports A-D and ADC; 'digital' for 'DIN'/'DOUT', 'aux' not implemented)

    Returns:
        np.ndarray -- Array with channel names present.
        Warning: channels are sorted. Do not use this output as a reference for the order of the channels.
    """
    block_to_names = {'analog': 'channel_names',
                      'digital': 'dig_channel_names'}

    all_chans = np.unique(
        np.hstack(meta_pd.loc[:, block_to_names[block]].values))
    if chan_filt.size > 0:
        found_stack = np.stack(
            [np.char.find(all_chans, s) == 0 for s in chan_filt])
        sel_chans = np.logical_or.reduce(found_stack)
        all_chans = all_chans[sel_chans]

    return all_chans


def find_chan_names_idx(all_chans: np.ndarray, chan_list: np.ndarray) -> np.ndarray:
    """ find the indices of an array of channel names

    Arguments:
        all_chans {np.ndarray} -- array of strings with all channel names
        chan_list {np.ndarray} -- array of strings with a selection of chan names

    Returns:
        np.ndarray -- array of channel indices
    """
    found_stack = np.stack(
        [np.char.find(all_chans, s) == 0 for s in chan_list])
    sel_chans = np.logical_or.reduce(found_stack)
    return np.where(sel_chans)[0]


def rec_chan_idx(meta_pd: pd.DataFrame, rec: int, chan_name_list: np.ndarray,
                 block: str='analog') -> np.ndarray:
    if block == 'analog':
        chan_name_filed = 'channel_names'
    elif block == 'digital':
        chan_name_filed = 'dig_channel_names'

    rec_all_chans = meta_pd.loc[meta_pd['name'] == 0,
                                chan_name_filed].values[0]
    found_stack = np.stack(
        [np.char.find(rec_all_chans, s) == 0 for s in chan_name_list])
    sel_chans = np.logical_or.reduce(found_stack)
    return np.where(sel_chans)[0]


def diff_array_columns(pd: pd.DataFrame, field: str, stride: int = 1) -> list:
    # return 'diff'  type comparison of a column containing a vector
    compared_list = [(x == y)
                     for (x, y) in zip(pd[field], pd[field].shift(stride))]
    return compared_list


def check_continuity(meta_pd: dict, chan_idx_list=np.empty(0), rec_list=np.empty(0), check_fileds: list = ['channel_names', 'channels_sample_rate']):

    rec_list = meta_pd['name'].values if rec_list.size == 0 else rec_list
    chan_idx_list = np.arange(
        meta_pd.loc[0, 'channel_names'].size) if chan_idx_list.size == 0 else chan_idx_list
    rec_slice = meta_pd['name'].isin(rec_list)

    continuity_dict = {}
    for col_name in check_fileds:
        comp_list = diff_array_columns(meta_pd.loc[rec_slice, :], col_name)

        # first column of comp_list is a scalar True
        comp_array = np.stack(comp_list[1:])
        # comp_array is shape [rec_slice.size - 1, n_chan (tota)]
        cont_recs = np.all(comp_array[:, chan_idx_list], axis=1)
        continuity_dict[col_name] = cont_recs

    cont_pd = pd.DataFrame(continuity_dict)
    cont_ok = True
    try:
        assert(cont_pd.all), 'Error in continuity of recordings'
    except:
        logger.warn(
            'There were differences in metadata of selected channnels and recs')
        logger.warn('{}'.format(cont_pd[~cont_pd.all(1)]))
        cont_ok = False
    finally:
        return cont_ok, cont_pd


@h5_decorator(default_mode='r')
def kwd_to_binary(kwd_file, out_file_path, chan_list: np.ndarray = np.empty(0),
                  rec_list: np.ndarray = np.empty(0),
                  chunk_size=8000000, header='bin'):
    """
    :param kwd_file: kwd file or kwd file path
    :param out_file_path: path to the bin file that will be created
    :param chan_list: list of channels. Default (empty) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    # header: 'bin' for flat binary, 'mda' for mountainsort
    # get the dataset of each recording and concatenateit to the out_file_path
    logger.info('Writing kwd_file {} to binary'.format(kwd_file.filename))

    # select recordings
    rec_list = get_rec_list(kwd_file) if rec_list.size == 0 else rec_list
    all_meta_pd = get_all_rec_meta(kwd_file)
    rec_slice = all_meta_pd['name'].isin(rec_list)

    # select channels
    if chan_list.size == 0:
        logger.debug('Will extract all channels')
        chan_list = get_all_chan_names(all_meta_pd)

    logger.info('Channels to extract: {}'.format(chan_list))

    # check continuity of channels across the list of recordings for the selected channels
    #cont_check_ok, cont_pd = check_continuity(sess_meta_pd, chan_idx_list, rec_list=rec_list)
    logger.info('Will go through recs {}'.format(rec_list))

    logger.info('Creating binary file {}'.format(out_file_path))
    total_samples = all_meta_pd.loc[rec_slice, 'samples_count'].sum()
    elements_in = 0
    with open(out_file_path, 'wb') as out_file:
        if header is 'mda':
            logger.debug('mda header')
            template_data = get_data_set(kwd_file,
                                         all_meta_pd.loc[rec_list[0], 'name'])  # get the first array to template the header
            data_type = np.dtype(template_data.dtype)

            mda_hdr = write_mda_hdr_explicit(
                len(chan_list), total_samples, data_type, out_file)

        pbar = tqdm((rec_list),
                    total=rec_list.size,
                    desc='{}'.format(os.path.split(out_file_path)[-1]),
                    leave=True)
        for rec_name in pbar:
            pbar.set_postfix(rec=' {}'.format(rec_name))
            rec_chans = all_meta_pd.loc[all_meta_pd['name']
                                        == rec_name, 'channel_names'].values
            assert rec_chans.size == 1, 'Either none or too many rec {} found'.format(
                rec_name)
            chan_numbers = find_chan_names_idx(rec_chans[0], chan_list)
            logger.debug('Chann numbers {}'.format(chan_numbers.tolist()))
            rec_elem = dset_to_binary_file(get_data_set(kwd_file, rec_name),
                                           out_file,
                                           chan_list=chan_numbers.tolist(),
                                           chunk_size=chunk_size,
                                           header=header
                                           )
            elements_in += rec_elem

    logger.info('{} elements written'.format(elements_in))
    return all_meta_pd.loc[rec_slice, :]


def kwd_append_recs(kwd_source: h5py.File, kwd_dest: h5py.File, rec_list: np.ndarray=np.empty(0)):
    # super basic, just copy groups
    if rec_list.size == 0:
        rec_list = get_rec_list(kwd_source)

    for rec in rec_list:
        kwd_source.copy('/recordings/{}'.format(rec), kwd_dest['/recordings'])


def update_kwd(kwd_path, rhx_path):
    # the whole metadata of the folder
    # get the contents of the folder, intan2kwik style
    logger.info('updating kwd file {} from folder {}'.format(
        kwd_path, rhx_path))
    board = which_board(rhx_path)
    rhx_meta = fu.get_rhd_pd(rhx_path, file_extension=board)

    # the metadata of the existing file and look for the last timestamp present
    current_meta_pd = get_all_rec_meta(kwd_path)
    last_timestamp_in = current_meta_pd.sort_values('start_time').iloc[-1]['start_time']

    # get the first timestamp after that wihch is a beginning of a rec
    rhx_meta = fu.get_rhd_pd(rhx_path, file_extension=board)
    new_tstamp_pd = rhx_meta.sort_values('t_stamp').loc[(rhx_meta['rec_break']) &
                                                              (rhx_meta['t_stamp'] > last_timestamp_in)]
    if new_tstamp_pd.empty:
        logger.info('No new files to add to the file')
        return None, new_tstamp_pd, rhx_meta

    first_new_timestamp = new_tstamp_pd.iloc[0]['t_stamp']
    new_rhx_pd = rhx_meta.loc[rhx_meta['t_stamp'] >= first_new_timestamp, :]
    
    logger.info('Will add {} new files'.format(new_rhx_pd.index.size))
    #return new_rhx_pd
    # go through a temporary file
    tmp_sess_dir = tempfile.mkdtemp()
    kwd_temp_path = os.path.join(tmp_sess_dir, os.path.split(kwd_path)[-1])
    logger.debug('tmp path {}'.format(kwd_temp_path))
    try:
        with h5py.File(kwd_temp_path, 'a') as kwd_temp_file, h5py.File(kwd_path, 'r+') as kwd_file:
            first_header = intan_to_kwd_multirec(kwd_temp_file, new_rhx_pd)

            logger.info('merging back to {}'.format(kwd_path))
            kwd_append_recs(kwd_temp_file, kwd_file)
    finally:
        logger.info('removing temp file')
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(tmp_sess_dir)
    return first_header, new_rhx_pd, rhx_meta
