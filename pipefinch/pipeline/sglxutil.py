# functions to read SpikeGLX neuropixels phase 3B (imec system) file structure, metadata, and data
# These are useful sources of documentation on reading these files
# Here's the meaning of some of the metadata https://github.com/JaneliaSciComp/JRCLUST/wiki/.meta-file.
# In particular, there is an explanation of what channels in the probe are used
# and where they are located in the block.
# More detailed meta here https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/Metadata.md.
import os
import sys
import glob
import logging
import configparser
import datetime
import parse
import shutil
import copy
import numpy as np
import pandas as pd
import warnings
import h5py
import contextlib

#from intan2kwik.core.h5 import tables
from pipefinch.h5tools.core import tables

from pipefinch.h5tools.core.h5tools import h5_unicode_hack
from pipefinch.pipeline import filestructure as et
from pipefinch.util import fileutil as fu
from intan2kwik import kwd

logger = logging.getLogger('pipefinch.pipeline.sglxutil')


def sgl_file_struct(sess_folder: str) -> (dict, pd.DataFrame):
    # make a dictionary with :
    # 'nidq'
    # 'imec_i' for every probe connected

    sess_name = os.path.split(sess_folder)[-1]
    folder_struct = {'nidq': sess_folder,
                     'imec': {}}
    files_pd = pd.DataFrame()

    # probes
    probe_paths = glob.glob(os.path.join(sess_folder, '*imec?'))
    probe_folders = [os.path.split(x)[-1] for x in probe_paths]
    probes = [int(x.split('imec')[-1]) for x in probe_folders]

    for i_probe, probe_path in zip(probes, probe_paths):
        folder_struct['imec'][i_probe] = probe_path

    return folder_struct, files_pd

def sgl_struct(sess_par: dict, epoch: str) -> dict:
    # locations of the folders for the epoch
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'])
    exp_struct['folders'] = {k: os.path.join(v, epoch)
                  for k, v in exp_struct['folders'].items()}

    update_files = ['kwd', 'kwe', 'mda_raw', 'bin_raw', 'kwik', 'par']
    updated_files_dict = {k: os.path.join(os.path.split(v)[0],
                                          epoch,
                                          os.path.split(v)[-1]) for k, v in exp_struct['files'].items() if k in update_files}
    
    exp_struct['files'].update(updated_files_dict)
    exp_struct['files']['kwik'] = os.path.join(os.path.split(exp_struct['files']['kwik'])[0],
                                               'sort_{}'.format(sess_par['sort']),
                                               os.path.split(exp_struct['files']['kwik'])[-1])
    
    #logger.info(updated_files_dict)
    
    return exp_struct

def get_data_meta_path(either_file_path: str) -> tuple:
    """get either a meta o rr binary file path and return both as a tuple

    Arguments:
        either_file_path {str} -- path of a meta/binary file   
    Returns:
        [type] -- (binary_path, meta_path)
    """
    file_stripped = '.'.join(either_file_path.split('.')[:-1])
    return tuple([file_stripped + ext for ext in ['.bin', '.meta']])


def read_headless_config(file_path: str, dummy_sec_name: str='root') -> configparser.ConfigParser:
    """read a meta file
    It is in the worst possible form: a pseudo 'ini' file without headers
    Here, I add a dummy section name ['root'] and read it using configparser
    Arguments:
        file_path {str} -- path to the .meta file (ini format without headers)

    Keyword Arguments:
        dummy_sec_name {str} -- name for the dummy header section (default: {'root'})

    Returns:
        configparser.ConfigParser -- ConfigParser object with the metadata
    """
    with open(file_path, 'r') as f:
        config_string = '[{}]\n'.format(dummy_sec_name) + f.read()
    config = configparser.ConfigParser()
    config.read_string(config_string)
    return config


def get_imec_meta(meta_file_path: str) -> dict:
    # get a dictionary with usually wanted metadata value: pairs
    config = read_headless_config(meta_file_path, dummy_sec_name='r')

    full_meta = dict(config['r'])  # read the full meta

    usual_meta = {'nsavedchans': int(config['r']['nsavedchans']),
                  's_f': float(config['r']['imsamprate']),
                  'meta_file_path': meta_file_path
                  }
    full_meta.update(usual_meta)
    return full_meta


def get_nidq_meta(meta_file_path: str) -> dict:
    # get a dictionary with usually wanted metadata value: pairs
    logger.info('meta file {}'.format(meta_file_path))
    config = read_headless_config(meta_file_path, dummy_sec_name='r')
    full_meta = dict(config['r'])  # read the full meta
    
    usual_meta = {'nsavedchans': int(config['r']['nSavedChans']),
                  's_f': float(config['r']['niSampRate']),
                  # n chans in multiplex neural, multiplex analog, direct analog, digital:
                  'sns_mn_ma_xa_dw': list(map(int, config['r']['snsMnMaXaDw'].split(','))),
                  'meta_file_path': meta_file_path
                  }
    full_meta.update(usual_meta)
    return full_meta


def get_imec_data(meta_file_path: str) -> dict:
    # can be a ap or a lf file
    # returns a dictionary
    bin_file_path = get_data_meta_path(meta_file_path)[0]
    imec_data = {'meta': None, 't_ms': None, 'neural': None, 'syn': None}

    # the meta
    imec_data['meta'] = get_imec_meta(meta_file_path)
    n_chan = imec_data['meta']['nsavedchans']  # nSavedChans in meta file
    s_f = imec_data['meta']['s_f']  # 30000.533148 #imSampleRate in meta file

    data_mmap = np.memmap(bin_file_path, dtype='int16',
                          mode='r').reshape(n_chan, -1, order='f')
    n_sample = data_mmap.shape[1]
    imec_data['t_ms'] = np.arange(n_sample)/s_f * 1000
    imec_data['neural'] = data_mmap
    imec_data['syn'] = data_mmap[-1]
    return imec_data


def get_ni_data(meta_file_path: str) -> dict:
    bin_file_path = get_data_meta_path(meta_file_path)[0]
    ni_data = {'meta': None, 't_ms': None, 'adc': None, 'dig_in': None}
    # read the analog channels
    # the meta
    ni_data['meta'] = get_nidq_meta(meta_file_path)

    n_chan_nidq = ni_data['meta']['nsavedchans']  # nSavedChans in meta file
    s_f_nidq = ni_data['meta']['s_f']  # nidqSampleRate in meta file
    n_chan_list = ni_data['meta']['sns_mn_ma_xa_dw']  # acqMnMaXaDw -
    # The two last are the relevant. We have no multiplexed channels (Whisper system);
    # those will always be zeros, or should be.
    n_adc_chan = n_chan_list[2]
    n_dig_word = n_chan_list[3]

    # logger.info(ni_data)
    # this is the whole block, including all four types of channels
    data_mmap_nidq = np.memmap(bin_file_path, dtype='int16', mode='r')
    data_mmap_nidq = data_mmap_nidq.reshape(n_chan_nidq, -1, order='f')
    n_sample_nidq = data_mmap_nidq.shape[1]
    # the time vector
    ni_data['t_ms'] = np.arange(n_sample_nidq)/s_f_nidq * 1000

    # read the analog inputs.
    ni_data['adc'] = data_mmap_nidq[:n_adc_chan, :]
    # this yields an array shaped [n_ch, n_sample]
    try:
        ni_data['dig_in'] = parse_digital(data_mmap_nidq, dig_words=n_dig_word)
    except NotImplementedError:
        warnings.warn(
            'Cant decode {} words of digital data yet'.format(n_dig_word))
    return ni_data


def parse_digital(data_mmap_nidq: np.ndarray, dig_words: int=1) -> np.ndarray:
    """ Parse the digital inputs from a nidq block. 
    The block can be a memmpapped array (see numpy memmap)
    The digital channels come in the last 2/4 byte of the file, which is read as int16. 
    As of june 2018 and for our system with no whisper and just the one PCI 6221 board,
    there is only 1 word of dig data (8 channels plus 8 empty). 
    using dig_words>1 will raise error, its there for future systems.

    Arguments:
        data_mmap_nidq {np.ndarray} -- [n_chan, n_sample] binary file with nidq data read as int16.
    Keyword Arguments:
        dig_words {int} -- (default: {1}). Number of words used for digital data.
                                        Valuse other than 1 will raise NotImplementedError 

    Returns:
        np.ndarray -- [n_dig_chan, n_sample] array with the digital channels ([dig_0, dig_1, ... order])
    """
    if not dig_words == 1:
        raise NotImplementedError
    # so get the digital channels
    # consider they are big endian to the extent of the .view() method.
    # In our case,there are 8 bits that matter (for the NI6221), but sglx uses entire bytes.
    # Hence, there is one whole byte with the DIN, which needs to be unpacked and only 8 bytes used.
    # Python has a sweet np.unpackbits() method for that, but only works with uint8.
    # So we need to cast it as a big_endian, 16 bit, and se the view() method to allow for np.uint8 input
    dig_stream = data_mmap_nidq[-1].astype(np.uint16)
    # unpack.
    # cast as big_endian, view as np.uint8 and reshape to [n_sample, n_chan]
    # (see accepted answer in https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types)
    all_dig_stream_16 = np.unpackbits(dig_stream.astype('>i2').view(np.uint8))
    # that one is a 1D array of 16 bits. we need to reshape and get rid of the 8 empty bits and reverse the order of those 8 bits,
    # so that dig[0, ... 7] coincide with x[:, 0...7]
    # order 'F' avoids having to transpose and the final array is [n_ch, n_sample]
    all_dig_stream = all_dig_stream_16.reshape(
        [8, 2, -1], order='f')[::-1, 1, :]
    return all_dig_stream


def datetime_from_sgl_meta(file_path: str) -> datetime.datetime:
    """get the creation time of a spikeglx meta file (same as the corresponding .bin files)
    Arguments:
        file_path {str} -- path to a .meta file ('ini' config file without sections)

    Returns:
        datetime.datetime -- time stamp of the recording (second precision)
    """
    sgl_meta = read_headless_config(file_path, dummy_sec_name='root')
    time_str = sgl_meta.get('root', 'fileCreateTime')
    return datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")


def read_chan_order(meta_parse) -> pd.DataFrame:
    # meta_parse is the ini parameter file, read with read_headless_config
    # we want a three col array with [ch_name, order_in_group, order_in_total]
    chans = meta_parse.get('root', '~snsChanMap')
    chans_tuple_list = chans.split(')(')[1:]
    chans_tuple_list[-1] = chans_tuple_list[-1].split(')')[0]
    all_tuples = [parse.parse('{};{}:{}', x).fixed for x in chans_tuple_list]
    return pd.DataFrame(all_tuples, columns=['name', 'order', 'abs_order'])


def quick_frames(data: np.array, stamps, samp_pre: int=-300, samp_post: int=3000) -> np.ndarray:
    # data [n_chan, n_samp]
    # return [nchan, n_stamp, n_sample_window]
    n_trigs = stamps.size
    samp_window = samp_post - samp_pre

    all_frames_idx = np.repeat(
        np.arange(samp_window), n_trigs).reshape(-1, n_trigs)
    all_frames_idx = all_frames_idx + samp_pre + stamps
    flat_frames = data[:, all_frames_idx.T.flatten()]
    # [n_ch, n_trig, n_sampl_window]
    frames_3d = flat_frames.reshape(-1, n_trigs,
                                    samp_post - samp_pre, order='c')
    return frames_3d


def create_nidq_grp(rec_grp, nidq_dict: dict, rec: int, include_channels: np.array=None, block: str='adc'):
    # creates a recording group with the adc from a nidq_dict obtained via get_ni_data
    logger.debug('Creating data group for this rec {}'.format(rec))

    ni_meta = nidq_dict['meta']
    ni_data = nidq_dict[block]

    # default is all channels in the block
    if include_channels is None:
        include_channels = np.arange(ni_data.shape[0])

    start_string = datetime.datetime.strptime(
        ni_meta['filecreatetime'], '%Y-%m-%dT%H:%M:%S')
    rec_start = start_string.strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')

    data_grp = rec_grp.require_group('{}'.format(rec))
    # append the metadata to this data group
    data_grp.attrs.create('bit_depth', 16, dtype=np.int16)
    data_grp.attrs.create('sample_rate', ni_meta['s_f'], dtype=np.float32)
    data_grp.attrs.create('name', rec)
    data_grp.attrs.create('start_sample', 0, dtype=np.int16)
    data_grp.attrs.create('start_time', rec_start)

    bit_volt = (int(ni_meta['niairangemax']) -
                int(ni_meta['niairangemin']))/32768

    all_chan_names = ['{}-{:02d}'.format(block, i) for i in include_channels]
    all_chan_names_uni = [h5_unicode_hack(x) for x in all_chan_names]
    n_chan = len(all_chan_names)
    all_bit_volts = np.ones(n_chan) * bit_volt
    all_rates = np.ones(n_chan) * ni_meta['s_f']

    # create application data
    #super_app_data_grp = data_grp.require_group('application_data')
    app_data_grp = data_grp.require_group('application_data')
    app_data_grp.attrs.create('is_multiSampleRate_data', 1, dtype=np.uint8)
    app_data_grp.attrs.create('channels_sample_rate', all_rates)
    app_data_grp.attrs.create('channel_bit_volts', all_bit_volts)
    app_data_grp.attrs.create('channel_names', all_chan_names_uni)
    return data_grp


def sgl_to_kwd_onerec(kwd_file, meta_file_path: str, rec: int, include_blocks: list) -> dict:
    # make the .kwd file
    # make the /recording/{} i group
    # dump header to application data
    # run rhd_rec_to table to create the table in the group

    logger.debug('Creating the /recordings/{} with {} files'.format(rec, 1))
    # if len(include_blocks)>1:
    #    raise NotImplementedError('Dont know how to handle multiple blocks yet')

    # create the rec group
    rec_grp = kwd_file.require_group('recordings')

    # get the data
    nidq_dict = get_ni_data(meta_file_path)
    for block in include_blocks:
        logger.info('block {}'.format(block))
        if block == 'adc':
            dset_name = 'data'
            data_grp = create_nidq_grp(
                rec_grp, nidq_dict, rec, include_channels=None, block=block)
            
            # create application_data for the metadata
            sgl_epoch = os.path.normpath(meta_file_path).split(os.path.sep)[-2]
            app_data_grp = data_grp.require_group('application_data')
            logger.info('created app_data_grp, will add sglx_epoch {}'.format(h5_unicode_hack(sgl_epoch)))
            app_data_grp.attrs.create('sglx_epoch', 
                                      h5_unicode_hack(sgl_epoch))
        else:
            dset_name = block

        data_block = nidq_dict[block].T
        t_block = nidq_dict['t_ms'].reshape([-1, 1])
        name_t = '{}-{}'.format('t_ms', block)
        # create the group and insert the tables
        data_type = data_block.dtype
        dset = tables.unlimited_rows_data(
            data_grp, dset_name, data_block.astype(data_type))
        #logger.info('t dataset {} has shape {}'.format(name_t, t_block.shape))
        tset = tables.unlimited_rows_data(data_grp, name_t, t_block)

        # add 'dig_channel_names' attribute to application_data group.
        if block == 'dig_in':
            include_channels = np.arange(data_block.shape[1])
            #logger.info('include channels dig {}'.format(include_channels))
            all_chan_names = ['DIN-{:02d}'.format(i) for i in include_channels]
            all_chan_names_uni = [h5_unicode_hack(x) for x in all_chan_names]
            app_data_grp = data_grp.require_group('application_data')
            app_data_grp.attrs.create('dig_channel_names', all_chan_names_uni)
            

            # make the 'events' detections (detect digital edges)
            # gets array t, chan_idx, edge
            # have to cast as short if wants to see falling edges!!
            dig_edge_arr = kwd.get_edges(nidq_dict[block].T.astype(np.short))
            edge_tables_dict = {}
            edge_tables_names = ['dig_edge_t', 'dig_edge_ch', 'dig_edge']
            for col, table in enumerate(edge_tables_names):
                col_array = dig_edge_arr[:, col].reshape([-1, 1])
                if col == 0:
                    col_array = dig_edge_arr[:, col].reshape([-1, 1])
                edge_tables_dict[table] = tables.unlimited_rows_data(data_grp, table,
                                                                     col_array)

    return nidq_dict


def sgl_to_kwd(meta_path, dest_file_path, rec=0, include_blocks=['adc', 'dig_in'], overwrite=False) -> dict:
    """
    :param meta_path: (string) path to the meta file of the recording
    :param dest_file_path: (string) dest of the kwd file
    :param rec: (int)
    :param include_blocks: (list of str) blocks to get
    :param overwrite: bool, whether to overwrite any pre-existing kwd file
    :return:
    """
    # make the .kwd file
    # make the /recording/0 group
    # dump header to application data
    # run rhd_rec_to table to create the table in the group

    if os.path.isfile(dest_file_path):
        logger.warn('File {} already exists'.format(dest_file_path))
        if overwrite:
            logger.info('Will create a new kwd file and overwrite the old one')
        else:
            logger.warn(
                'Will abort. To overwrite the existing file, use overwrite=True option')
            raise RuntimeError('Kwd file already exists')

    logger.info('dest file: {}'.format(dest_file_path))
    # deletes
    try:
        os.makedirs(os.path.split(dest_file_path)[0], exist_ok=True)
        with h5py.File(dest_file_path, 'w') as kwd_temp_file:
            data_dict = sgl_to_kwd_onerec(
                kwd_temp_file, meta_path, rec, include_blocks)
    except:
        logger.info('Failed to complete kwd file; removing it')
        with contextlib.suppress(FileNotFoundError):
            os.remove(dest_file_path)
        raise
    return data_dict


def all_sgl_to_kwd(sess_par: dict, include_blocks=['adc', 'dig_in'], overwrite=False):
    # points to all the raw data in a folder and get the blocks to kwd
    all_sess_folders = list_sgl_epochs(sess_par, raw_paths=True)
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'], sess_par['sort'])
    kwik_folder = exp_struct['folders']['kwik']
    raw_folder = exp_struct['folders']['raw']

    logger.info(
        'will process to kwd all epochs in session folder skipping trouble sessions {}'.format(raw_folder))
    logger.info('found {} epoch subfolders'.format(len(all_sess_folders)))
    trouble_sessions = []
    for sess_folder in all_sess_folders:
        logger.info('epoch folder {}'.format(sess_folder))
        try:
            sgl_folder, sgl_pd = sgl_file_struct(sess_folder)
            nidq_meta_files = glob.glob(os.path.join(sgl_folder['nidq'], '*.meta'))
            nidq_meta_file = nidq_meta_files[0]

            #dest_file_name = os.path.split(nidq_meta_file)[-1].split('.nidq.meta')[0] + '.kwd.nidq'
            dest_file_name = 'stream.kwd'
            dest_file_path = os.path.join(kwik_folder, os.path.split(
                sgl_folder['nidq'])[-1], dest_file_name)
       
            nidq_dict = sgl_to_kwd(nidq_meta_file, dest_file_path,
                               rec=0, include_blocks=include_blocks, overwrite=True)
        except BaseException as e:
            logger.warning('Error in sess {}'.format(sess_folder))
            warnings.warn('Error in sess {}: {}'.format(sess_folder, str(e)))
            trouble_sessions.append(sess_folder)

    logger.info('Done with all')
    return trouble_sessions


def list_sgl_epochs(sess_par: dict, raw_paths=False) -> list:
    # points to all the epochs in a session
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'], sess_par['sort'])

    kwik_folder = exp_struct['folders']['kwik']
    raw_folder = exp_struct['folders']['raw']

    all_sess_folders = list(
        filter(os.path.isdir, glob.glob(os.path.join(raw_folder, '*'))))
    all_sess_folders.sort()
    if raw_paths:
        return all_sess_folders
    else:
        return list(map(lambda x: os.path.split(x)[-1], all_sess_folders))

### tools to merge epochs

def merge_epochs(sess_par: dict, epochs_to_merge: list, overwrite: bool=True) -> tuple:
    logger.info('Will merge epochs {}'.format(epochs_to_merge))
    new_epoch = '{}_{}'.format(sess_par['sess'], 'alles')
    epoch_structs = [sgl_struct(sess_par, epoch) for epoch in epochs_to_merge]
    epoch_folders = [sgl_file_struct(epoch_struct['folders']['raw'])[0] for epoch_struct in epoch_structs]
    exp_struct = sgl_struct(sess_par, new_epoch)
    sgl_folder, _ = sgl_file_struct(exp_struct['folders']['raw'])
    logger.info('SGL folder struct {}'.format(sgl_folder))
    
    # make de folders
    merged_raw_folder = exp_struct['folders']['raw']
    logger.info('Creating merged raw {}'.format(merged_raw_folder))
    if overwrite:
        logger.info('Will cleanup destination folder first: {}'.format(merged_raw_folder))
        if os.path.exists(merged_raw_folder):
            shutil.rmtree(merged_raw_folder)
    os.makedirs(merged_raw_folder, exist_ok=True)
    
    # copy set files from the first session
    # Note that it is empty
    copy_raw = []
    for k, v in exp_struct['files'].items():
        if k in copy_raw:
            #print(v)
            shutil.copyfile(epoch_structs[0]['files'][k], v)
            
    # copy the raw_metadatas
    epoch_meta_list = []
    for epoch_folder in epoch_folders:
        epoch_meta = copy.deepcopy(epoch_folder)
        logger.info('* Adding epoch {}'.format(epoch_folder['nidq']))
        for k, v in epoch_folder.items():
            if isinstance(v, dict):
                # deal with the probes
                for prb, epoch_fold in v.items():
                    src_fold = epoch_fold
                    probe_folder = '{}_imec{}'.format(new_epoch, prb)
                    dst_path = os.path.join(merged_raw_folder, probe_folder)
                    src_meta_paths = merge_raw_sgl(src_fold, dst_path) # yields lf, ap meta
                    
                    logger.info('imec meta paths {}'.format(src_meta_paths))
                    for band in ['ap', 'lf']:
                        meta_path = [x for x in src_meta_paths if band in x][0]
                        full_meta = get_imec_meta(meta_path)
                        epoch_meta[k + '_meta_' + band] = {prb: full_meta}
                #os.makedirs(dest_fold[0], exist_ok=True)
            else: #it's the nidaq
                src_fold = v
                dst_path = merged_raw_folder
                src_meta_paths = merge_raw_sgl(src_fold, dst_path) # now it's only one meta path, 
                # merge_raw_sgl will throw error if there are more than one trigger
                # get the meta
                full_meta = get_nidq_meta(src_meta_paths[0])
                # append it to the dict
                epoch_meta[k + '_meta'] = full_meta
        
        epoch_meta_list.append(epoch_meta)
                
    return exp_struct, epoch_structs, epoch_meta_list, new_epoch



def merge_raw_sgl(src_fold: str, dest_fold: str) -> list:
    logger.info('merging {} into {}'.format(src_fold, dest_fold))
    os.makedirs(dest_fold, exist_ok=True)
    meta_files = glob.glob(os.path.join(src_fold, '*.meta'))
    # check that there is only one triggered recording in the folder (only one tuple of bin, meta with t0 as identifier)
    t_instances = np.unique([int(x.split('_')[-1].split('.')[0].strip('t')) for x in meta_files])
    n_trigs = t_instances.size
    if n_trigs != 1:
        if n_trigs > 1:
            raise NotImplementedError('Many triggers (t0, t1,...) in the epoch, dont know how to handle yet')
        else:
            raise RuntimeError('No recording t identifiers found (t0)')       
    logger.info(meta_files)
    
    # copy each meta_file to the corresponding, only if there is none of the class (i.e, leave just the first meta file for each merged session)
    new_base = os.path.split(dest_fold)[-1]
    if new_base.find('imec') >=0:
        new_base, _, prb = new_base.rpartition('_')
    logger.info('new base {}'.format(new_base))
    for m_path in meta_files: # for now it's only one
        meta_file = os.path.split(m_path)[-1]
        m_fname, m_ext = meta_file.split('.', 1)
        m_base, _,  m_t = m_fname.rpartition('_')
        new_meta_name = '{}_{}'.format(new_base, m_t)
        new_meta_file = '{}.{}'.format(new_meta_name, m_ext)
        logger.info('New meta is {}'.format(new_meta_file))
        logger.info('dest_fold {}'.format(dest_fold))
        # check if dest exists and copy
        new_bin_path, new_meta_path = get_data_meta_path(os.path.join(dest_fold, new_meta_file))
        fu.safe_copy(m_path, new_meta_path)
        
        # append binaries
        bin_path, _ = get_data_meta_path(m_path)
        fu.append_binary(bin_path, new_bin_path)
    return meta_files

def make_merged_kwd(merged_exp_struct: dict, epoch_meta_list: list, overwrite=False):
## use these epoch folders to read the metas and make the kwd of the merged thing
    ## get the metas for all the nidq in the sorted epoch folders
    logger.info('merging nidq data of {} epochs onto kwd file {}'.format(len(epoch_meta_list), merged_exp_struct['files']['kwd']))
    
    ni_pd = pd.DataFrame([x['nidq_meta'] for x in epoch_meta_list])
    ni_pd['filetimesecs'] = ni_pd['filetimesecs'].apply(np.float)
    ni_pd['samples'] = ni_pd['filetimesecs'] * ni_pd['s_f']
    
    ni_pd.sort_values('filecreatetime')
    logger.info('Epochs to merge {}'.format(ni_pd['meta_file_path']))
    
    # signal the breaking points within the concatenated
    ni_pd['starts'] = ni_pd['samples'].shift(1)
    ni_pd.loc[0, 'starts'] = 0
   
    ## make the kwd with sgl_to_kwd(meta_path, dest_file_path, rec=0, include_blocks=['adc', 'dig_in'], overwrite=False) -> dict
    kwd_path = merged_exp_struct['files']['kwd']
    
    meta_folder = sgl_file_struct(merged_exp_struct['folders']['raw'])[0]['nidq']
    nidq_meta_files = glob.glob(os.path.join(meta_folder, '*.meta'))
    nidq_meta_path = nidq_meta_files[0]
    logger.info('nidq meta path {}'.format(nidq_meta_path))
    merged_nidq_dict = sgl_to_kwd(nidq_meta_path, kwd_path, overwrite=overwrite)
    
    ## edit the kwd and add the metadata on the breaks to /application_data tables
    with h5py.File(kwd_path, 'r+') as kwd_file:
        app_data_group = kwd_file['/recordings/0/application_data']
        
        tables.insert_table(app_data_group, ni_pd['starts'].to_numpy(dtype=np.int), 'breaks_sample')
        tables.insert_table(app_data_group, ni_pd['filecreatetime'].to_numpy(), 'breaks_tstart', force_dtype=h5py.special_dtype(vlen=str))
        tables.insert_table(app_data_group, ni_pd['meta_file_path'].to_numpy(), 'breaks_file', force_dtype=h5py.special_dtype(vlen=str))
        tables.insert_table(app_data_group, ni_pd['samples'].to_numpy(dtype=np.int), 'breaks_file_samples')
    
    return merged_nidq_dict, ni_pd