import numpy as np
import logging
import glob
import os

#from intan2kwik.core.h5 import tables
from pipefinch.neural.convert.mdautil import update_mda_hdr, write_mda_hdr, mda_fun_dict
from intan2kwik.core import reading
from intan2kwik import kwd as ikwd

logger = logging.getLogger('pipefinch.neural.convert')


def intan_to_mda(all_rhx_files: list, dest_file_path: str, dtype: str=np.int16) -> tuple:
    """
    Converts a list of rh{d,s} files to one .mda file, creating an intermediate plain binary file.
    Can handle both files from the rhd and rhs system; only extracts the neural channels.
    Arguments:
        all_rhx_files {list} -- list of paths to files to read and concatenate
        dest_file_path {str} -- destination file path

    Keyword Arguments:
        dtype {str} -- datatype to use (np.int16 or np.float typycally) (default: {np.int16})

    Raises:
        NotImplementedError -- still can't handle dtypes other than int16/32 and float32

    Returns:
        dict, int -- a dictionary with the first header of the list of intan files
    """    
    logger.info('Reading {} intan files of into mda file {}'.format(len(all_rhx_files), dest_file_path))
    # make intermediate binary file
    
    dest_folder, dest_file = os.path.split(dest_file_path)
    bin_file_path = os.path.join(dest_folder, dest_file.split('.')[0] + '.bin')
    logger.debug('Making intermediate binary file (c-style) {}'.format(bin_file_path))
    intan_meta, samples_in = intan_to_bin(all_rhx_files, bin_file_path, include_channels=['amplifier'], dtype=dtype)
    n_neural_chans = intan_meta['num_amplifier_channels']
    logger.debug('Should have {} channels, {} samples'.format(n_neural_chans, samples_in))
    
    # read the binary to mmap and save it as the final file
    x = np.memmap(bin_file_path, dtype=dtype).reshape(-1, n_neural_chans)
    logger.info('Writing to {} mda file'.format(dest_file_path))
    
    if dtype == np.int16:
        dt = 'int16'
    elif dtype == np.int32:
        dt = 'int32'
    elif dtype == np.float32:
        dt = 'float32'
    else:
        raise NotImplementedError
    
    mda_fun_dict[dt](x.T, dest_file_path)
    os.remove(bin_file_path)
    return intan_meta, samples_in

def rhd_rec_to_dict(rhx_file_path: list, chan_groups_wishlist: list=['amplifier']) -> tuple:
    """[summary]
    Arguments:
        rhx_file_path {list} -- (list of strings) paths of files to include in this rec

    Keyword Arguments:
        chan_groups_wishlist {list} -- (flat ndarray/list) channel groups to get from the rhd files (default: {['amplifier']})

    Returns:
        tuple -- dictionary of np. arrays [n_samples, n_chans] with entry for every chan group
    """ 
    # make the table
    # read the blocks and append them to the table
    # set the attributes of the table
    read_block = reading.read_data(rhx_file_path)

    # filter include groups, warn if a group wasn't in the data and remove it from the list
    include_chan_groups = []
    block_data = {}
    block_t = {}
    for g in chan_groups_wishlist:
        if '{}_data'.format(g) in read_block.keys():
            include_chan_groups.append(g)
        else:
            logger.warn('Channel data group {} is not in the recordings'.format(g))
    logger.debug('include groups {}'.format(include_chan_groups))
    # The main data (neural chans and dac chans; from include_chans)
    for ch_grp in include_chan_groups:
        block_data[ch_grp] = (read_block['{}_data'.format(ch_grp)].astype(np.int32) - 32768).T
        block_t[ch_grp] = (read_block['t_{}'.format(ch_grp)].astype(np.int32) - 32768).T
    
    # The digital channels
    try:
        block_data['dig_in'] = read_block['board_dig_in_data'].T.astype(np.short)
        block_t['dig'] = read_block['t_dig'].reshape([-1, 1])
    except KeyError:
        logger.warn('No digital channels')

    return block_data, block_t

def intan_to_bin(all_rhx_files:list, dest_file_path: str, \
                 include_channels: list=['amplifier'], \
                 board: str='auto', \
                 dtype: str=np.int16, variant: str='bin', order:str='C') -> tuple:
    
    """
    Converts a list of rh{d,s} files to one .mda file, creating an intermediate plain binary file.
    Can handle both files from the rhd and rhs system.
    All files in the list should have similar header (number of channels, sample_rate)
    It can do different orders (column minor 'C'  column major 'F')
    It can dump directly onto an mda file (variant='mda')

    :param all_rhx_files: (list) folder where the .rh? files are
    :param dest_file_path: (string) dest of the kw? files
    :param include_channels: (flat ndarray/list) (see intan2kwik documentation)
    :param board: (str) 'rhd' or 'rhs' for rhd2000 or rhs2000, 'auto' for detecting from extension
    :param dtype (str): data type desired (defalut np.int16)
    :param variant (str): 'bin' (plain binary) or 'mda' (mda with header and column major order)
    :param order (str): 'C' (column minor) or 'F' (column major) ordering in the binary file
    :return: dict, int -- Dictionary with header of first file
    """
        
    # attributes from the header
    first_header = reading.read_intan_header(all_rhx_files[0])
    # logger.info('First header {}'.format(first_header))
    v_multipliers = [0.195, 50.354e-6]
    if first_header['eval_board_mode'] == 1:
        v_multipliers[1] = 152.59e-6
        
    # list all the chunk files:
    samples_in = 0
    logger.debug('File type is {}'.format(variant))
    
    with open(dest_file_path, 'wb') as out_file:
        logger.info('Creating the output file {}'.format(dest_file_path))
        
        for i, rhx_file_path in enumerate(all_rhx_files):
            logger.info('File {}/{}'.format(i, len(all_rhx_files)))
            data_dict, t_dict = rhd_rec_to_dict(rhx_file_path, include_channels)
            amp_data = data_dict['amplifier'].astype(dtype)
            # amp_data comes in [n_samp, n_ch] shape

            # Check what to do if is mda
            if variant == 'mda':
                logger.debug('Target is mda file, overrriding order to Fortran')
                order = 'C'
                if i == 0:
                    logger.debug('is mda')
                    # write a 'dummy mda_header with the right number of dimensions'
                    # but the wrong number of elements
                    hdr = write_mda_hdr(amp_data.T, out_file)
                out_file.write(amp_data.reshape((amp_data.shape[1], amp_data.shape[0]), order=order).tobytes(order=order))
                    
            if variant=='bin':
                out_file.write(amp_data.tobytes(order=order))
            # for the .bin, c_style, this appends chunks to a [n_samp, n_ch]
            samples_in += data_dict['amplifier'].shape[0]
        
        if variant == 'mda':
            hdr[3] = first_header['num_amplifier_channels']
            hdr[4] = samples_in
            update_mda_hdr(hdr, out_file)
    
    return first_header, samples_in

def folder_to_files(folder: str, board: str='auto') -> str:
    """
    List with full path of all the rh{d, s} files in a folder, sorted.
    Arguments:
        folder {str} -- path containing intan recordings

    Keyword Arguments:
        board {str} -- [Board ('rhs', 'rhd', 'auto') (default: {'auto'})

    Returns:
        str -- list of full paths of the files
    """
    logger.info('reading intan chans data across all of rec {0}'.format(folder))
    if board=='auto':
        board = ikwd.which_board(folder)
    all_rhx_files = glob.glob(os.path.join(folder, '*.{}'.format(board)))
    all_rhx_files.sort()
    logger.debug('Found {} .{} files to process'.format(len(all_rhx_files), board))
    return all_rhx_files