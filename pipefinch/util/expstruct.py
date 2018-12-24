import errno
import glob
import logging
import os
import socket

# for more than structure
import numpy as np
import yaml
#todo: let go of yaml and use .json

from numpy.lib import recfunctions as rf

from pipefinch.util.file import file_functions as ff

logger = logging.getLogger('bci.core.expstruct')


# file structure

def get_definitions_dictionaries():
    # packages is a dictionary with {'description': 'path relative to repos folder'}
    packages = {'sound': 'soundflow',
                'ephys': 'ephys_flow',
                'analysis': 'analysis_tools',
                'swiss': 'swissknife'}

    locations = {'zebra':
                     {'repos': os.path.abspath('/home/zeke/repos'),
                      'experiment': os.path.abspath('/data/experiment'),
                      'experiment_local': os.path.abspath('/data/experiment'),
                      'archive': os.path.abspath('/mnt/uberfinch/carbon/archive/bci_zf'),
                      'scratch': os.path.abspath('/home/zeke/scratch')},
                 }

    return {'packages': packages,
            'locations': locations}


def get_computer_name():
    return socket.gethostname()


def get_locations(dict_path=None, comp_name=None):
    if dict_path is None:
        if comp_name is None:
            comp_name = get_computer_name()
            locations = get_definitions_dictionaries()['locations'][comp_name]
    else:
        #
        raise NotImplementedError('Still dont know how to load a dictionary of locations')
    return locations


def set_paths(repos_root=None):
    if repos_root is None:
        repos_root
    pass


def is_none(x):
    return 'none' == str(x).lower()


def flex_file_names(bird, sess='', rec=0, experiment_folder=None, base='experiment', location='experiment'):
    fn = file_names(bird, sess, rec, experiment_folder, base)

    exp_base = fn['locations'][location]
    folders = {'raw': os.path.join(exp_base, 'raw_data', bird, sess),  # local raw
           'ss': os.path.join(exp_base, 'ss_data', bird, sess),
           'rw': os.path.join(exp_base, 'raw_data', bird, sess),  # stored raw
           'stim': os.path.join(exp_base, 'stim_data', bird, sess),
           'tmp': os.path.join(exp_base, 'tmp_data', bird, sess),
           'templ': os.path.join(exp_base, 'templates'),
           'prb': os.path.join(exp_base, 'probes')}

    fn['folders'] = folders
    return fn


def read_yml(file_path):
    with open(file_path, 'r') as f:
        contents = yaml.load(f)
    return contents

def file_names(bird, sess='', rec=0, experiment_folder=None, base='experiment'):
    computer_name = get_computer_name()

    if experiment_folder is None:
        experiment_folder = get_definitions_dictionaries()['locations'][computer_name]['experiment']
        experiment_local = get_definitions_dictionaries()['locations'][computer_name]['experiment_local']
        base_location = get_definitions_dictionaries()['locations'][computer_name]
    else:
        experiment_local = experiment_folder
        base_location = {'experiment': os.path.abspath(experiment_folder),
                         'experiment_local': os.path.abspath(experiment_folder),
                         'store': os.path.abspath(experiment_folder),
                         'archive': os.path.abspath(experiment_folder)}

    folders = {'raw': os.path.join(experiment_local, 'raw_data', bird, sess),  # local raw
               'ss': os.path.join(experiment_folder, 'ss_data', bird, sess),
               'rw': os.path.join(experiment_folder, 'raw_data', bird, sess),  # stored raw
               'proc': os.path.join(experiment_folder, 'proc_data', bird, sess), #processed data
               'stim': os.path.join(experiment_folder, 'stim_data', bird, sess),
               'tmp': os.path.join(experiment_local, 'tmp_data', bird, sess),
               'templ': os.path.join(experiment_folder, 'templates'),
               'prb': os.path.join(experiment_folder, 'probes'),
               'kai': os.path.join(os.path.abspath('/mnt/cube/kai/results'), bird, sess)}


    files = {'structure': base,
             'ss_raw': base + '.raw.kwd',
             'ss_lfp': base + '.lfp.kwd',
             'ss_bin': base + '.dat',
             'ss_par': base + '.par.yml',
             'par': base + '.par.yml',
             'sng': base + '.sng.kwe',
             'stm': base + '.stm.kwe',
             'cand': base + '.mot.h5',
             'evt': base.split('_')[0] + '.kwe',
             'mic': base + '-rec_{0:03}.mic.wav'.format(int(rec)),
             'sts': base + '-rec_{0:03}.sts.wav'.format(int(rec)),
             'kk_prb': '*.prb',
             'kk_par': 'params.prm',
             'ks_par': 'params.py',
             'ks_mas': 'master.m',
             'ks_map': 'chanMap.mat',
             'ks_cfg': 'config.m'}

    return {'folders': folders,
            'structure': files,
            'locations': base_location}


def file_path(fn_dict, folder_key, file_key):
    """
    :param fn_dict: dictionary of file_names (as output of file_names)
    :param folder_key: string, key to folders (folder type)
    :param file_key: string, key to structure (file type)
    :return:
    """
    return os.path.join(fn_dict['folders'][folder_key], fn_dict['structure'][file_key])


def mkdir_p(path):
    logger.debug('Creating directory {}'.format(path))
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            logger.debug('Directory existed, doing nothing')
            pass
        else:
            raise


def sub_dirs(path):
    return [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]


def list_birds(folder, breed='z'):
    all_dirs = [os.path.split(d)[-1] for d in sub_dirs(folder)]
    all_birds = [b for b in all_dirs if b.startswith(breed)]
    all_birds.sort()
    return all_birds

def list_sessions(bird, experiment_folder=None, location='ss'):
    fn = file_names(bird, experiment_folder=experiment_folder)
    bird_folder = fn['folders'][location]
    sessions_bird = [s for s in os.listdir(bird_folder) if os.path.isdir(os.path.join(bird_folder, s))]
    return sessions_bird


def list_raw_sessions(bird, sess_day=None, depth='', experiment_folder=None, location='raw'):
    all_sessions = list_sessions(bird, experiment_folder=experiment_folder, location=location)
    if sess_day is not None:
        all_sessions = [s for s in all_sessions if sess_day in s]
    if depth != '':
        all_sessions = [s for s in all_sessions if int(s[0].split('_')[-1]) == int(depth)]
    all_depths = ['{}'.format(s.split('_')[-1]) for s in all_sessions]
    return all_sessions, all_depths


# Experiment structure
def get_parameters(bird, sess, rec=0, experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, rec, experiment_folder=experiment_folder)
    pars = read_yml(os.path.join(fn['folders'][location], fn['structure']['par']))
    return pars


def update_parameters(new_par, bird, sess, rec=0, experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, rec, experiment_folder=experiment_folder)
    par_file_path = os.path.join(fn['folders'][location], fn['structure']['par'])
    bkp_path = ff.make_backup(par_file_path)
    logger.info('Overwriting parameter file; backup in {}'.format(bkp_path))
    with open(par_file_path, 'w') as f:
        written = yaml.dump(new_par, f)
    return written


def get_stims_dict(bird, sess, rec=0, experiment_folder=None, location='ss'):
    """
    get the dictionary of stimuli {name: file_name.wav}
    :param bird:
    :param sess:
    :param rec:
    :param experiment_folder:
    :param location:
    :return: dict
    """
    exp_pars = get_parameters(bird, sess,
                              rec=rec,
                              experiment_folder=experiment_folder,
                              location=location)
    return exp_pars['search_motiff']['motiff_patterns']


def stim_id(exp_par, name):
    """
    Get a stimulus name's id in the .evt file of events (its group name)
    :param exp_par: parameters of the experiment (read .yml file)
    :param name: the name of the stimulus in the 'motiff_patterns' dictionary
    :return: the id of the parameter
    """
    patterns = exp_par['search_motiff']['motiff_patterns']
    return patterns[name].split('.wav')[0]