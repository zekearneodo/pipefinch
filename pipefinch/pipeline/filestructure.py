import os
import socket

locations_dict = dict()
locations_dict['zebra'] = {'mnt': os.path.abspath('/data/experiment'),
                           'local': os.path.abspath('/data/experiment')}
locations_dict['ZOROASTRO'] = {'mnt': os.path.abspath('B:\microdrive\data'),
                               'local': os.path.abspath('D:\microdrive')}
locations_dict['lookfar'] = {'mnt': os.path.abspath('/Users/zeke/experiment'),
                             'local': os.path.abspath('/Users/zeke/experiment')}
# Zinch in windows
locations_dict['Zinch'] = {'mnt': '\\\\ZUPERFINCHJR\storage\Data',
                           'local': os.path.abspath('C:\experiment')}

# Zinch in linux
locations_dict['zinch'] = {'mnt': '/mnt/zuperfinchjr/Data',
                           'local': os.path.abspath('/media/zinch/Windows/experiment')}

default_struct_par = {'neural': 'ephys',
                      'presort': 'kwik',
                      'sort': 'msort'}
# Todo: define locations_dict in a json file or something more elegant


def get_locations_from_hostname():
    hostname = socket.gethostname()
    return locations_dict[hostname]


def read_json_exp_struct():
    raise NotImplementedError


def get_file_structure(location: dict, sess_par: dict, struct_par: dict = default_struct_par) -> dict:
    exp_struct = {}
    bird, sess = sess_par['bird'], sess_par['sess']

    exp_struct['folders'] = {}
    exp_struct['files'] = {}

   # The raw files
    exp_struct['folders']['raw'] = os.path.join(
        location['mnt'], bird, 'ephys', 'raw', sess)
    for f, n in zip(['par', 'set', 'rig'],
                    ['experiment.json', 'settings.isf', 'rig.json']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['raw'], n)

    # the kwik system (spikes, events, kwd file with streams)
    exp_struct['folders']['kwik'] = os.path.join(
        location['local'], bird, 'ephys', 'kwik', sess)
    for f, n in zip(['kwd', 'kwik', 'kwe'], ['streams.kwd', 'spikes.kwik', 'events.kwe']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['kwik'], n)

    # the aux, temporary mountainsort files. these will be deleted after sorting
    exp_struct['folders']['msort'] = os.path.join(
        location['local'], bird, 'ephys', 'msort', sess)
    for f, n in zip(['mda_raw', 'par'], ['raw.mda', 'params.json']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['msort'], n)

    return exp_struct


def get_exp_struct(bird, sess, location_dict: dict = dict()):
    # get the configuration of the experiment:
    # if environment variable 'EXPERIMENT_PATH' exists,
    # read 'EXPERIMENT_PATH/config/expstruct.json'
    # no location dict was entered, try to get it
    if location_dict:
        pass
    else:
        read_exp_base = os.environ.get('EXPERIMENT_PATH')

        if read_exp_base is not None:
            # if there is a .json file configured with the variables of the experiment
            exp_base = os.path.abspath(read_exp_base)
            location_dict_json_path = os.path.join(exp_base, 'exp_struct.json')
            location_dict = read_json_exp_struct()
        else:
            # try to read it from the hostname
            location_dict = get_locations_from_hostname()

    # make the exp struct dict.
    sess_par_dict = {'bird': bird,
                     'sess': sess}
    exp_struct = get_file_structure(location_dict, sess_par_dict)

    return exp_struct
