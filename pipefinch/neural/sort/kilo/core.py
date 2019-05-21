import os
import logging
import subprocess
import numpy as np
import scipy.io as sio
from string import Template

from pipefinch.pipeline import probes
from pipefinch.neural.sort import kilo

logger = logging.getLogger('pipefinch.neural.sort.kilo.core')

default_kilo_pars = {'kilo_version': 2,  # version of kilosort to use (1 or 2)
                     'n_filt': None,  # total n of filters (multiple of 32)
                     'filt_per_chan': 4,  # filters per channel
                     'auto_merge': 1,  # automerge
                     'use_gpu': 1,  # must be 1 for Kilosort 2
                     'port': 0,
                     # These must be given, there are no valid defaults
                     'n_chan': None,  # total number of chans in the .bin file
                     's_f': None,  # sampling rate
                     # these are default pars for phy. can be changed
                     'offset': 0,
                     'hp_filtered': True,
                     'dtype_name': np.int16.__name__,
                     }

default_kilo_paths = {
    # paths for Kilosort2
    2: {'kilo_dir': os.path.abspath(
        '/home/ezequiel/repos/Kilosort2'),
        'npymat_dir': os.path.abspath(
        '/home/ezequiel/repos/npy-matlab'),
        'template_dir': None},
    # paths for KiloSort
    1: {'kilo_dir': os.path.abspath(
        '/home/ezequiel/repos/KiloSort'),
        'npymat_dir': os.path.abspath(
        '/home/ezequiel/repos/npy-matlab'),
        'template_dir': None}
}


def make_paths(ss_folder: str, out_subfolder_name: str='sort_out') -> dict:
    """
    Makes all the paths of the files of the mountainlab sort pipeline
    Arguments:
        ss_folder {str} -- folder with the source (raw.mda, params.json, [geom.csv]) files

    Keyword Arguments:
        out_subfolder_name {str} -- name of the output folder (will be a subfolder of ss_folder) (default: {'sort_out'})

    Returns:
        dict -- dictionary with file_pathd necessary for the pipeline
    """

    input_f_names = {'bin': 'raw.bin',
                     'params': 'params.json',
                     'prb': 'chanMap.mat'}

    output_f_names = {'rez': 'rez2.mat',  # kilosort output file
                      'mat_log': 'kilosort_mat.log',  # matlab log file,
                      'phy_par': 'params.py'
                      }

    #out_folder_path = os.path.join(ss_folder, out_subfolder_name)
    out_folder_path = ss_folder
    file_paths = {k: os.path.join(ss_folder, v)
                  for k, v in input_f_names.items()}
    file_paths.update({k: os.path.join(out_folder_path, v)
                       for k, v in output_f_names.items()})
    return file_paths, out_folder_path


def get_templates(kilo_paths: dict, kilo_version: int=2) -> dict:
    """[summary]
    Get the templates for kilosort matlab files master and config for replacment with this experiment's parameters
    Arguments:
        kilo_paths {dict} -- dictionary of kilosort paths. needs template_dir, pointing to the directory where the templates 
        (master.tpl, config.tpl)are.
        If kilo_paths['template_dir'] is None, it will look for the default templates that come with the repository and are in 
        /pipefinch/neural/sort/kilo
        kilo_version {int} -- version of kilosort (1/2)
    Returns:
        dict -- Dictionary with Template objects (returned when reading the files master.tpl, config.tpl):
        tpl_dict = {'master': Template,
                    'config': Template,
                    'run_master: Template}
    """
    if kilo_paths[kilo_version]['template_dir'] is None:
        kilo_paths[kilo_version]['template_dir'] = os.path.join(os.path.split(kilo.__file__)[0], 'templates')

    logger.debug('Getting the kilosort {} master.m and config.m templates from {}'.format(kilo_version,
                                                                                          kilo_paths[kilo_version]['template_dir']))
    tpl_dict = {}

    master_template = 'master.tpl.{}'.format(kilo_version)
    config_template = 'config.tpl.{}'.format(kilo_version)
    run_master_template = 'run_master.tpl'
    with open(os.path.join(kilo_paths[kilo_version]['template_dir'], master_template), 'r') as src:
        tpl_dict['master'] = Template(src.read())
    with open(os.path.join(kilo_paths[kilo_version]['template_dir'], config_template), 'r') as src:
        tpl_dict['config'] = Template(src.read())
    with open(os.path.join(kilo_paths[kilo_version]['template_dir'], run_master_template), 'r') as src:
        tpl_dict['run_master'] = Template(src.read())
    return tpl_dict


def do_the_sort(kilo_paths: dict):
    logger.info('Running kilosort on matlab')
    raw_file = kilo_paths['bin']
    sort_folder = os.path.split(raw_file)[0]
    logger.info('Sort folder is {}'.format(sort_folder))
    log_file = kilo_paths['mat_log']
    # run_master exits mlab after running master with code 0 (success), -1 (fail)
    mlb_cmd = '-r "cd(\'{}\'); dir; run_master"'.format(sort_folder) 
    log_cmd = '-logfile {}'.format(log_file)
    logger.debug('Issue command {}'.format(mlb_cmd))
    logger.info('output to {}'.format(log_file))

    # pause the module logger and log to a different thing.
    logger.disabled = True
    sorter = subprocess.check_output(
        ['matlab', '-nodesktop', '-nosplash', '-noawt', mlb_cmd, log_cmd])
    logger.disabled = False
    sorter_output = str(sorter, 'utf8')
    return sorter_output, int(float(sorter_output.split('return_value=')[-1]))


def make_kilo_scripts(local_sort_dir, user_kilo_pars: dict = {}, user_kilo_paths: dict = {}):
    # update the kilo_paths with any user-entered values (if any)
    kilo_paths = default_kilo_paths
    kilo_paths.update(user_kilo_paths)
    # update the kilo_pars with any user_entered values (if any)
    kilo_pars = default_kilo_pars
    kilo_pars.update(user_kilo_pars)
    # resolve n_filts:
    # if auto_merge is false
    # If n_filt was not specified, use filt_per chan to calculate.
    # according to ksort documentation, needs to be a multiple of 32
    # Kilosort2 ignores this parameter, but Kilosort (version 1) uses it for the initial sorting
    if kilo_pars['n_filt'] is None:
        n_filt = int(np.ceil(kilo_pars['n_chan']
                             * kilo_pars['filt_per_chan']/32)*32)
        kilo_pars['n_filt'] = n_filt

    logger.debug('local kilosort dir: {}'.format(local_sort_dir))

    kilo_version = user_kilo_pars['kilo_version']
    version_kilo_paths = kilo_paths[kilo_version]
    # parameters are all for version 1/2; the templates take care of the rest
    subst_par = {
        'kilo_dir': version_kilo_paths['kilo_dir'],
        'npy_matdir': version_kilo_paths['npymat_dir'],
        'data_dir': local_sort_dir,
        's_f': kilo_pars['s_f'],
        'n_chan': kilo_pars['n_chan'],
        'n_filt': kilo_pars['n_filt'],
        'auto_merge': kilo_pars['auto_merge'],
        'use_gpu': kilo_pars['use_gpu']
    }
    logger.debug('Substituting template parameters with {}'.format(subst_par))

    # get the Templates from the template files
    tpl_dict = get_templates(kilo_paths, kilo_version=kilo_version)

    # substitute templates with these parameters
    for name, template in tpl_dict.items():
        script_path = os.path.join(local_sort_dir, name + '.m')
        with open(script_path, 'w') as f:
            f.write(template.substitute(subst_par))
            logger.info('Written kilo script {}'.format(script_path))


def make_kilo_chanmap(prb_dict: dict, chan_map_path: str) -> dict:
    # see https://github.com/cortex-lab/KiloSort
    # count the total number of channels
    n_channels = len(probes.flatten_probe(prb_dict).keys())

    # make all the chanMap arrays for kilosort
    # all channels are 'connected' or 'active' (filter happens in prb_dict)
    connected = np.array([True] * n_channels).reshape((n_channels, 1))

    chan_map = np.arange(n_channels) + 1
    chan_map_zero_ind = np.arange(n_channels)

    x_coords = np.ones((n_channels, 1)) * -1
    y_coords = np.ones((n_channels, 1)) * -1
    k_coords = np.ones((n_channels, 1)) * -1

    for i_group, (grp_name, grp_dict) in enumerate(prb_dict.items()):
        for channel, coord_tuple in grp_dict.items():
            x_coords[channel], y_coords[channel] = coord_tuple
            k_coords[channel] = i_group + 1

    chan_map_dict = {'name': 'pipefinch.neural.kilo.core.make_kilo_chanmap',
                     'Nchannels': n_channels,
                     'connected': connected,
                     'chanMap': chan_map,
                     'chanMap0ind': chan_map_zero_ind,
                     'xcoords': x_coords,
                     'ycoords': y_coords,
                     'kcoords': k_coords}

    sio.savemat(chan_map_path, chan_map_dict)
    return chan_map_dict


def make_phy_par_file(user_kilo_pars: dict, file_paths: dict) -> dict:
    # make the parameters file for phy
    # https://phy-contrib.readthedocs.io/en/latest/template-gui/
    kilo_pars = default_kilo_pars
    kilo_pars.update(user_kilo_pars)
    phy_pars = {'dat_path': file_paths['bin'],
                'n_channels_dat': kilo_pars['n_chan'],
                'offset': kilo_pars['offset'],
                'dtype': kilo_pars['dtype_name'],
                'sample_rate': kilo_pars['s_f'],
                'hp_filtered': kilo_pars['hp_filtered']
                }
    # now turn these into a python file

    with open(file_paths['phy_par'], 'w+') as f:
        [f.write('{} = {}\n'.format(k, repr(v))) for k, v in phy_pars.items()]
    logger.info('Written phy parameters file {}'.format(file_paths['phy_par']))
    return phy_pars
