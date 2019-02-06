# Common functions for kwd/kwik/kwe files (structure of recordings)
import logging
import shutil
import os
import glob
import datetime


import h5py
import numpy as np
from scipy import signal as ss
from scipy.io import wavfile

from pipefinch.h5tools.core.h5tools import h5_decorator, h5_unicode_hack

module_logger = logging.getLogger("pipefinch.h5tools.kwik.kutil")

## Other util
def parse_tstamp(t_string: str):
    # t_string comes in 'yyyy-mm-dd hh:mm:ss'
    # convert to timestamp
    d, t = t_string.split(' ')
    t_stamp = datetime.datetime(*tuple(map(int, d.split('-') + t.split(':'))))
    return t_stamp

# K file utils
@h5_decorator(default_mode='r')
def get_rec_list(k_file):
    """
    :param k_file: (kwik or kwd)
    :return: list of recordings in an h5file (kwik/kwd) as a sorted numpy array
    """
    rec_list = list(map(int, k_file['/recordings'].keys()))
    #module_logger.info('rec list {}'.format(rec_list))
    return np.sort(rec_list)


@h5_decorator(default_mode='r')
def get_rec_attrs(h5, recording):
    return h5['/recordings/{}'.format(int(recording))].attrs


@h5_decorator(default_mode='r')
def get_rec_start_sample(h5, recording):
    return get_rec_attrs(h5, recording)['start_sample']


@h5_decorator(default_mode='r')
def get_record_sampling_frequency(h5, recording=0):
    path = 'recordings/{0:d}'.format(recording)
    return h5[path].attrs.get('sample_rate')


@h5_decorator(default_mode='r')
def get_rec_group(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: h5 group
    """
    logging.debug('Getting group of rec {}'.format(rec))
    return kwd_file['/recordings/{}'.format(int(rec))]


@h5_decorator(default_mode='r')
def get_rec_origin(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: dictionary with bird, sess, rec
    """
    group = get_rec_group(kwd_file, rec)
    origin_strings = h5_unicode_hack(group.attrs.get('name')).split(':')
    recording = int(origin_strings[1].split('/')[-1])
    path = os.path.split(origin_strings[0])[0]
    base_name = os.path.split(origin_strings[0])[-1].split('.')[0]
    sess = os.path.split(path)[-1]
    bird = os.path.split(os.path.split(path)[0])[-1]
    return {'bird': bird, 'sess': sess, 'rec': recording, 'structure': base_name}


@h5_decorator(default_mode='r')
def rec_start_array(kwik):
    rec_list = list(map(int, get_rec_list(kwik)))
    rec_array = np.arange(max(rec_list) + 1)
    start_array = np.zeros_like(rec_array)
    for i_rec in rec_list:
        start_array[i_rec] = get_rec_start_sample(kwik, i_rec)
    return start_array


@h5_decorator(default_mode='r')
def get_corresponding_rec(kwik, stamps):
    '''
    Get the vector of recordings to which an array of absolute stamps belong
    :param kwik: kwik_file (open in r mode), to get the rec starts
    :param stamps: a list of stamps, absolute, refering to beginning of rec 0
    :return: recs
    '''
    rec_list = get_rec_list(kwik)
    rec_starts = rec_start_array(kwik)
    stamps_recs = np.array(
        [rec_list[np.where(rec_starts < start)[0][-1]] for start in stamps])
    return stamps_recs

@h5_decorator(default_mode='r')
def apply_rec_offset(h5, stamp_array, rec_array):
    rec_offsets = rec_start_array(h5)
    return apply_offsets(stamp_array, rec_array, rec_offsets)

@h5_decorator(default_mode='r')
def apply_offsets(stamps, recs, offset_array):
    offsets = list(map(lambda i: offset_array[i], recs))
    return np.array(stamps) + np.array(offsets)
