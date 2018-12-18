import logging
import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rf

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
    logging.debug('Getting dataset from rec {}'.format(rec))
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
    all_rec_meta_list = list(map(lambda rec: get_rec_attrs(kwd_file, rec), all_rec_list))
    all_rec_meta_dict = {k: [r[k] for r in all_rec_meta]}
    # flatten it and toss it in a pandas dataframe
    apd = pd.concat([pd.DataFrame([r[k] for r in all_rec_meta_dict]) for k in all_rec_meta_dict[0].keys()], axis=1)
    #todo: refine the dict
    return apd

@h5_decorator(default_mode='r')
def kwd_to_binary(kwd_file, out_file_path, chan_list=None, rec_list=[], chunk_size=8000000):
    """
    :param kwd_file: kwd file or kwd file
    :param out_file_path: path to the bin file that will be created
    :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    # get the dataset of each recording and concatenateit to the out_file_path
    logging.info('Writing kwd_file {} to binary'.format(kwd_file.filename))
    
    # select channels
    if chan_list is not None:
        if (type(chan_list) is not list) and (type(chan_list) is not tuple):
            assert (type(chan_list) is int)
            chan_list = [chan_list]
        chan_list = list(chan_list)
    logging.info('Channels to extract: {}'.format(chan_list))

    # select recordings
    rec_list = get_rec_list(kwd_file) if rec_list == [] else rec_list
    logging.info('Will go through recs {}'.format(rec_list))
    
    logging.info('Creating binary file {}'.format(out_file_path))
    elements_in = 0
    with open(out_file_path, 'wb') as out_file:
        for rec_name in rec_list:
            rec_elem = dset_to_binary_file(get_data_set(kwd_file, rec_name),
                                                                   out_file,
                                                                   chan_list=chan_list,
                                                                   chunk_size=chunk_size
                                                                   )
            elements_in += rec_elem
    
    logging.info('{} elements written'.format(elements_in))