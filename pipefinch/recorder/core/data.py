# tools to read data files from recorder (.lvd)
import struct
from datetime import datetime, timedelta
from io import IOBase
import warnings
import logging
import numpy as np
import os
import pandas as pd

logger = logging.getLogger('pipefinch.recorder.core.data')

# File utils


def open_file_decorator(open_mode='rb'):
    # Decorator to open a file with mode parameters, if a string is entered,
    # or work with the file pointer, if a IOBase is entered
    def open_file_wrapper_mode_set(func):
        # Decorator without parameters.
        # Note that open_mode is defined by the outside decorator function
        def wraped_func(file_or_path, *args, **kwargs):
            #print('file path {}'.format(file_or_path))
            if isinstance(file_or_path, str):
                with open(file_or_path, open_mode) as fid:
                    return_vaule = func(fid, *args, **kwargs)
            elif isinstance(file_or_path, IOBase):
                return_vaule = func(file_or_path, *args, **kwargs)
            else:
                raise ValueError('function {} Received neither string, nor open file'.format(
                    func.__name__))
            return return_vaule
        return wraped_func
    return open_file_wrapper_mode_set


def meta_path_decorator(func):
    # Decorator to point to the meta file if a data file is enter
    def wraped_func(file_path, *args, **kwargs):
        #print('file path {}'.format(file_or_path))
        file_ext = os.path.split(file_path)[-1].split('.')[-1]
        if file_ext == 'txt':
            return_value = func(file_path, *args, **kwargs)
        elif file_ext == 'lvd':
            file_path = get_meta_file_path(file_path)
            return_value = func(file_path, *args, **kwargs)
        else:
            raise ValueError('function {} Received neither lvd, nor meta file path'.format(
                func.__name__))
        return return_value
    return wraped_func


def get_file_size(fid: IOBase):
    fid.seek(0, 2)
    size_bytes = fid.tell()
    return size_bytes


@open_file_decorator(open_mode='rb')  # Decorate so it can take a file_name
def read_header(fid: IOBase, style: str='chronic') -> tuple:
    if style is 'chronic':
        # the header for the 'chronic' mode is a set of 4 doubles:
        # [sampling_rate, n_channels, time_stamps, dynamic_range]
        # I got this from the ll_header function in matlab.
        # It is big endian ('ieee-be' in matlab)
        # read 4 doubles
        # NOTE: there are more channels in the file than the chanels I could count in the
        # channel_info entry in the corresponding settings.txt files.
        # I don't understand what I am missing, or why the hell someone would save channels into a
        # binary file that contains data and not include those channels in the fucking settings file
        # that is supposed to describe the settings of the software that is making said file.
        # In any case, that is happening in extra channels that come after those described in the settings,
        # to the best of my knowledge. I am just ingoring them but carrying them.
        hdr_size = 32
        fid.seek(0)
        hdr_tuple = struct.unpack('>dddd', fid.read(hdr_size))
    else:
        raise NotImplementedError
    return hdr_tuple + (hdr_size, )


def set_data_type(file_format: int, style: str='chronic') -> str:
    # configure the data type of the recording
    if style is 'chronic':
        # int16, bigendian
        labview_type = '>h'
    else:
        raise NotImplementedError('Dont know datatpe other than chronic 1 yet')
    return labview_type


def dict_hdr(hdr_tuple):
    hdr_dict = {'hdr_size': hdr_tuple[4],
                'n_channels': int(hdr_tuple[1]),
                'sample_rate': hdr_tuple[0],
                'dynamic_range': hdr_tuple[3]}
    return hdr_dict


def read_file(file_path: str, style: str='chronic', file_format: int=1) -> tuple:
    data_type = set_data_type(file_format, style)
    elem_byte_size = struct.calcsize(data_type)

    with open(file_path, 'rb') as fid:
        hdr = read_header(file_path, style=style)
        file_size = get_file_size(fid)

        hdr_size_bytes = hdr[4]
        data_block_bytes = file_size - hdr_size_bytes

        n_channels = int(hdr[1])
        n_samples = data_block_bytes//(n_channels * elem_byte_size)
        fid.seek(hdr_size_bytes)
        data = np.fromfile(fid, dtype=data_type)
    
    data = data.reshape((n_samples, n_channels))
    hdr_dict = dict_hdr(hdr)
    meta_dict = read_meta_file(file_path)
    
    return data, hdr_dict, meta_dict

def file_as_data_frame(file_path) -> pd.DataFrame:
    day_file, day_hdr, day_meta = read_file(file_path)
    data_df = pd.DataFrame(day_file)
    for k,v in day_meta.items():
        data_df[k] = v
    data_df['file'] = os.path.split(file_path)[-1].split('.')[0]
    
    # make absolute timestamps
    # s_f = day_hdr['sample_rate']
    # t_deltas = [timedelta(seconds=x) for x in data_df.index.values/s_f]
    # data_df['abs_tstamp'] = data_df['time_stamp'] + pd.Series(t_deltas)
    data_df['file_sample'] = data_df.index
    data_df['time_stamp'] = pd.to_datetime(day_meta['time_stamp'])
    return data_df

def fill_abs_tstamp(data_df: pd.DataFrame, s_f=32000) -> pd.DataFrame:
    t_deltas = [timedelta(seconds=x) for x in data_df['file_sample'].values/s_f]
    data_df['abs_tstamp'] = data_df['time_stamp'] + pd.Series(t_deltas)
    return data_df

def get_meta_file_path(data_file_path: str) -> str:
    folder, data_name = os.path.split(data_file_path)
    meta_name, meta_ext = data_name.split('.')
    return os.path.join(folder, 'Meta', meta_name + '.txt')

def process_meta_lines(text_lines: list) -> dict:
    # very crude, just get filetype and timestamp
    meta_dict = {}
    lines_arr = np.array([z.split('\n')[0].strip() for z in text_lines])
    meta_dict['file_type'] = lines_arr[np.where(lines_arr=='filetype:')[0] + 2][0]

    t_stamp = lines_arr[np.where(lines_arr=='*')[0] + 1][0]
    t_obj = datetime.strptime(t_stamp, '%Y%m%d%H%M%S.%f')
    meta_dict['time_stamp'] = t_obj
    return meta_dict
    

@meta_path_decorator
def read_meta_file(file_path: str) -> dict:
    #logger.info('reading meta from {}'.format(file_path))
    with open(file_path, 'r') as fid:
        file_lines = fid.readlines()
    #logger.info(file_lines)
    return process_meta_lines(file_lines)