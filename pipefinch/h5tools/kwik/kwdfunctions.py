import logging
import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rf

from pipefinch.neural.convert.mdautil import update_mda_hdr, write_mda_hdr_explicit, mda_fun_dict
from pipefinch.h5tools.core.h5tools import h5_decorator, list_subgroups, obj_attrs_2_dict_translator
from pipefinch.h5tools.core.tables import dset_to_binary_file
from pipefinch.h5tools.kwik.kutil import get_rec_list

logger = logging.getLogger('pipefinch.h5tools.kwik.kwdfunctions')


@h5_decorator(default_mode='r')
def get_data_set(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: h5 dataset object with
    """
    #logger.debug('Getting dataset from rec {}'.format(rec))
    return kwd_file['/recordings/{}/data'.format(int(rec))]


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
    rec_sizes = {i: get_data_size(kwd_file, rec_list[i])
                 for i in range(0, rec_list.size)}
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
def get_all_rec_meta(kwd_file) -> pd.DataFrame:
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
    return all_meta_pd


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
def kwd_to_binary(kwd_file, out_file_path, chan_list=None, rec_list=[], chunk_size=8000000, header='bin'):
    """
    :param kwd_file: kwd file or kwd file
    :param out_file_path: path to the bin file that will be created
    :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    # header: 'bin' for flat binary, 'mda' for mountainsort
    # get the dataset of each recording and concatenateit to the out_file_path
    logger.info('Writing kwd_file {} to binary'.format(kwd_file.filename))

    # select channels
    if chan_list is not None:
        if (type(chan_list) is not list) and (type(chan_list) is not tuple):
            assert (type(chan_list) is int)
            chan_list = [chan_list]
        chan_list = list(chan_list)
    logger.info('Channels to extract: {}'.format(chan_list))

    # select recordings
    rec_list = get_rec_list(kwd_file) if rec_list == [] else rec_list
    all_meta_pd = get_all_rec_meta(kwd_file)
    rec_slice = all_meta_pd['name'].isin(rec_list)

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
            data_type = template_data.dtpye()

            mda_hdr = write_mda_hdr_explicit(len(chan_list), total_samples, data_type, out_file) 
            
        for rec_name in rec_list:
            rec_elem = dset_to_binary_file(get_data_set(kwd_file, rec_name),
                                           out_file,
                                           chan_list=chan_list,
                                           chunk_size=chunk_size,
                                           header=header
                                           )
            elements_in += rec_elem

    logger.info('{} elements written'.format(elements_in))
