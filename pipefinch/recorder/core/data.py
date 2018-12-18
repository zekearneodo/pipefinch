# tools to read data files from recorder (.lvd)
import struct
from io import IOBase
import warnings
import logging
import numpy as np

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


def make_chan_info_dict(settings_dict: dict) -> dict:
    ch_info_tuple = tuple()
    for ch_id, ch_info_string in zip(settings_dict['channels'], settings_dict['channel info']):
        ch_info_list = ch_info_string.split(',')
        ch_info_dict = {'chan_id': ch_id,
                        'name': ch_info_list[1].strip(),
                        'save': ch_info_list[2].strip().split(' ')[1],
                        'gain': float(ch_info_list[3].strip().split(' ')[1]),
                        'rig': ch_info_list[0].strip()}
        ch_info_tuple += (ch_info_dict, )
    settings_dict['chan_info'] = ch_info_tuple
    return settings_dict


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

    data = np.memmap(file_path, dtype=data_type, mode='r',
                     offset=hdr_size_bytes, shape=(n_samples, n_channels))
    hdr_dict = dict_hdr(hdr)
    return data, hdr_dict
