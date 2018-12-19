import numpy as np
import logging
from mountainlab_pytools import mdaio
from io import IOBase

logger = logging.getLogger('pipefinch.neural.mdautil')

# Mountainsort header related dictionaries
# functions for writing for the different data types
mda_fun_dict = {'int16': mdaio.writemda16i,
                'int32': mdaio.writemda32i,
                'float16': mdaio.writemda,
                'float32': mdaio.writemda32}

# codes in the header for the different data types
mda_dt_codes = {'float32': -3,
               'int16': -4,
               'int32': -5,
               'uint16': -6,
               'double': -7,
               'uint32':-8}

def update_mda_hdr(hdr: np.array, file_ptr):
    """
    Update the header in an mda file
    Arguments:
        hdr {np.array} -- header (array of at least 4 int32)
        file_ptr {file_ptr} -- pointer to an open binary file
    """
    file_ptr.seek(0)
    file_ptr.write(hdr.astype(np.int32).tobytes())
    #print(hdr)

def write_mda_hdr(x: np.array, file_ptr) -> np.array:
    """""
    Generate the header of an mda file, to be filled with the array x, and write it.
    Arguments:
        x {np.array} -- array that will fill the file [n_ch, n_samples]
        file_ptr {file_pointer} -- pointer to an open ('wb') file

    Returns:
        np.array -- the header
    """
    dtype = x.dtype
    hdr = np.array([mda_dt_codes[dtype.name], dtype.itemsize, x.ndim])
    hdr = np.append(hdr, x.shape)
    file_ptr.write(hdr.astype(np.int32).tobytes())
    logger.debug('Created and written header {}'.format(hdr))    
    return hdr


def write_mda_hdr_explicit(n_ch, n_samples, dtype, file_ptr, n_dim=2) -> np.array:

    """""
    Generate the header of an mda file, to be filled with the array x, and write it.
    Arguments:
        x {np.array} -- array that will fill the file [n_ch, n_samples]
        file_ptr {file_pointer} -- pointer to an open ('wb') file

    Returns:
        np.array -- the header
    """
    hdr = np.array([mda_dt_codes[dtype.name], dtype.itemsize, n_dim, n_ch, n_samples])
    file_ptr.write(hdr.astype(np.int32).tobytes())
    logger.debug('Created and written header {}'.format(hdr))    
    return hdr
