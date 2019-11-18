# Set of functions for appending data and reformatting the kwik file (and kwe)
import logging
import shutil
import os
import glob
import json
import tqdm

import h5py
import numpy as np
import pandas as pd
import warnings
from numba import jit
from scipy import signal as ss
from scipy.io import wavfile

from pipefinch.h5tools.core import h5tools as h5t
from pipefinch.h5tools.kwik import kutil
from pipefinch.h5tools.kwik import kwdfunctions as kwdf
from pipefinch.h5tools.kwik import mdaio

module_logger = logging.getLogger("pipefinch.h5tools.kwik.kwikfunctions")


# List all the units in a file
@h5t.h5_decorator(default_mode='r')
def list_units(kf, group=0, sorted=False):
    # get the unit group
    qlt_path = "/channel_groups/{0:d}/clusters/main".format(group)

    g_dtype = np.int
    clu_dtype = np.int
    qlt_dtype = np.int

    clu_list = list(kf[qlt_path].keys())
    qlt_list = [
        kf["{0:s}/{1:s}".format(qlt_path, c)].attrs.get('cluster_group') for c in clu_list]
    n_spikes = len(clu_list)
    clu_dt = np.dtype(
        [('group', g_dtype, 1), ('clu', clu_dtype, 1), ('qlt', qlt_dtype, 1)])
    clu = np.recarray(n_spikes, dtype=clu_dt)

    clu['group'] = group
    clu['clu'] = np.array(list(map(int, clu_list)))
    clu['qlt'] = np.array(list(map(int, qlt_list)))

    if sorted:
        clu = clu[(clu['qlt'] == 1) | (clu['qlt'] == 2)]
    return clu


# List all the stimuli in a file
def list_sound_stimuli(h5, stim_table_path='/event_types/Stimulus/text'):
    datagroup = h5[stim_table_path]
    all_stim = np.array([s for s in datagroup[:] if not is_number(s)])
    return np.unique(all_stim)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def attrs2dict(node):
    return {key: val for key, val in node.attrs.items()}


def dict2attrs(meta_dict, node):
    if meta_dict is not None:
        assert node
        for key, val in meta_dict.items():
            node.attrs.create(key, val)

# These should come from h5t, but need to see what's the deal with append_attributes.
# Here it takes a list, in h5t it takes a dictionary!!!


def insert_group(parent_group, name, attr_dict_list=None):
    new_group = parent_group.create_group(name)
    if attr_dict_list is not None:
        append_atrributes(new_group, attr_dict_list)
    return new_group


def append_atrributes(h5obj, attr_dict_list):
    for attr_dict in attr_dict_list:
        # print attr_dict['name'] + ' {0} - {1}'.format(attr_dict['data'], attr_dict['dtype'])
        h5obj.attrs.create(
            attr_dict['name'], attr_dict['data'], dtype=attr_dict['dtype'])
        #h5obj.attrs.create(attr['name'], attr['data'], dtype=attr['dtype'])

@h5t.h5_decorator(default_mode='r')
def get_rec_attrs(kwik_file, rec) -> dict:
    rec_attrs = dict()
    # names of metadata groups and locations in the rec group:
    fields_locs = {'rec_group': '/recordings/{}'.format(rec)}

    for name, location in fields_locs.items():
        rec_attrs[name] = kwdf.obj_attrs_2_dict_translator(kwik_file[location])
    return rec_attrs

class KwikFileWriter:
    file_names = dict()
    kwd_path = ''
    kwik_path = ''
    chan_group = 0
    rec_sizes = np.empty(0)
    rec_start_samples = np.empty(0)
    
    # recs that were included in the sorted binary file
    rec_in_bin = np.empty(0)

    spk = np.empty(0)  # spikes tstamps (samples) 1d array
    clu = np.empty(0)  # clusters id matching spk, 1d array
    grp = tuple()  # tuple of (clu, sorted_descriptor)

    def __init__(self, file_names: dict, chan_group: int = 0, 
                 rec_in_binary: np.ndarray = np.empty(0),
                 raw_format = 'kwd'):
        
        self.file_names = file_names
        self.chan_group = chan_group
        
        self.kwik_path = file_names['kwik']
        self.rec_kwik = None
        self.spk_kwik = None
        self.kwf = None
        self.chan_group = chan_group
        # with open(file_names['par']) as f:
        #     exec (f.read())
        #     self.s_f = sample_rate
        
        # if data comes from rh, kwik and there is a kwd with neural data
        if raw_format == 'kwd':
            self.kwd_path = file_names['kwd']
            self.s_f = kutil.get_record_sampling_frequency(self.kwd_path)
            # fill the self.rec_sizes values
            self.get_rec_sizes_in_bin()
            self.rec_in_bin = rec_in_binary
            self.rec_start_samples = kwdf.get_rec_starts(self.file_names['kwd'])
            
        # if data comes from spikeGLX
        elif raw_format == 'sgl':    
            self.kwd_path = None
            self.rec_in_bin = np.array([0])
            self.rec_start_samples = np.array([0])
            self.rec_sizes = {0: np.iinfo(np.int64).max} # all is rec0
            self.s_f_from_params()
            
        self.create_kwf()

    def create_kwf(self):
        with h5py.File(self.file_names['kwik'], 'w') as kwf:
            kwf.require_group('/channel_groups')
            kwf.require_group('/recordings')

    def get_clusters(self):
        # needs doing in approrpiate class
        # here be set:
        # self.gru
        # self.clu
        # self.spk
        raise NotImplementedError

    def s_f_from_params(self):
        with open(self.file_names['params'], 'r') as jfile:
            ks_params = json.load(jfile)
        module_logger.info(ks_params['s_f'])
        self.s_f = ks_params['s_f']
        
    def get_rec_sizes_in_bin(self):
        # get the rec sizes
        # filter all the ones who are present in the binary file
        with h5py.File(self.kwd_path, 'r') as kwd:
            rec_sizes = kwdf.get_rec_sizes(kwd)
        if self.rec_in_bin.size == 0:
            self.rec_sizes = rec_sizes
        else:
            self.rec_sizes = {k: rec_sizes[k] for k in self.rec_in_bin}

    def make_spk_tables(self, realign_to_recordings=True):
        # needs to have filled
        # self.spk
        # self.clu

        # refer to starts, counting from the first present rec in the binary
        self.rec_kwik, self.spk_kwik = ref_to_rec_starts(self.rec_sizes,
                                                         self.spk)

        module_logger.info(self.s_f)
        with h5py.File(self.kwik_path, 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group(
                '{}'.format(self.chan_group))
            spikes_group = chan_group.require_group('spikes')
            h5t.insert_table(
                spikes_group, self.rec_kwik.flatten(), 'recording')
            if realign_to_recordings:
                h5t.insert_table(
                    spikes_group, self.spk_kwik.flatten(), 'time_samples')
                h5t.insert_table(
                    spikes_group, self.spk_kwik.flatten() / self.s_f, 'time_fractional')
            else:
                h5t.insert_table(
                    spikes_group, self.spk.flatten(), 'time_samples')

            clusters_group = spikes_group.require_group('clusters')
            h5t.insert_table(clusters_group, self.clu, 'main')
            h5t.insert_table(clusters_group, self.clu, 'original')

    def make_rec_groups(self):
        rec_list = np.unique(self.rec_kwik)
        rec_start_samples = self.rec_start_samples
        # module_logger.debug(rec_start_samples)
        #module_logger.info("Found recs {}".format(rec_list))
        with h5py.File(self.file_names['kwik'], 'r+') as kwf:
            rec_group = kwf.require_group('recordings')
            for rec in rec_list:
                #module_logger.info("table for rec {}".format(rec))

                rec_name = 'recording_{}'.format(rec)
                # module_logger.debug(rec_start_samples)
                # module_logger.info(rec_name)
                attribs = [{'name': 'name', 'data': rec_name, 'dtype': 'S{}'.format(len(rec_name))},
                           {'name': 'sample_rate', 'data': self.s_f,
                               'dtype': np.dtype(np.float64)},
                           {'name': 'start_sample',
                               'data': rec_start_samples[rec], 'dtype': np.int64},
                           {'name': 'start_time', 'data': rec_start_samples[rec] / self.s_f, 'dtype': np.float64}]
                #module_logger.info('Will make rec group for rec {}'.format(rec))
                try:
                    insert_group(rec_group, str(rec), attribs)
                # except ValueError as err:
                #     if 'Name already exists' in err.args[0]:
                #         module_logger.info('rec group already existed, skipping')
                #     else:
                #         raise
                except RuntimeError as err:
                    if 'Name already exists' in err.args[0]:
                        module_logger.info(
                            'rec group already existed, skipping')
                    else:
                        raise

    def make_clu_groups(self, name='main'):
        clu_grp_dict = {'mua': 1,
                        'noise': 0,
                        'unsorted': 3,
                        'accepted': 2,
                        'rejected': 9,
                        'artifact': 5
                        }

        with open(self.file_names['cluster_metrics'], 'r') as f:
            all_metrics = json.load(f)['clusters']
            # loads a list of dictionaries with keys ['label', 'metrics']
        accepted_clu = np.unique(self.clu)
        accepted_metrics = [
            m for m in all_metrics if m['label'] in accepted_clu]

        with h5py.File(self.file_names['kwik'], 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group(
                '{}'.format(self.chan_group))
            clusters_group = chan_group.require_group('clusters')
            desc_group = clusters_group.require_group(name)

            for metric in accepted_metrics:
                #clu_type = [x[1] for x in self.grp if x[0] == metric['label']]
                #module_logger.info('metrics {}'.format(metric['metrics']))
                # if there are tags in the json file, use them
                # otherwise default is 'unsorted'
                try:
                    clu_type = metric['tags'][0]
                except IndexError:
                    clu_type = 'unsorted'
                    metric['tags'] = ['unsorted']

                attribs = [{'name': 'cluster_group',
                            'data': clu_grp_dict[clu_type],
                            'dtype': np.int64}]

                this_cluster_group = insert_group(
                    desc_group, str(metric['label']), attribs)
                h5t.append_atrributes(this_cluster_group, metric['metrics'])
                tags_list_utf8 = [h5t.h5_unicode_hack(
                    x) for x in metric['tags']]
                h5t.append_atrributes(this_cluster_group,
                                      {'tags': tags_list_utf8})


class MdaKwikWriter(KwikFileWriter):
    mda_params = dict()
    # init the KwikFile class

    def __init__(self, file_names, chan_group=0, rec_in_binary=np.empty(0)):
        super(MdaKwikWriter, self).__init__(file_names, chan_group=chan_group,
                                            rec_in_binary=rec_in_binary,
                                            raw_format = 'kwd')

    def get_clusters(self):
        spk_data = mdaio.readmda(self.file_names['firings'])
        self.spk = spk_data[1, :].astype(np.int64)
        self.clu = spk_data[2, :].astype(np.int)
        self.grp = [(i, 'unsorted') for i in np.unique(self.clu)]

        # TODO: include cluster metrics (.json file)

    def make_clu_groups(self, name='main'):
        clu_grp_dict = {'mua': 1,
                        'noise': 0,
                        'unsorted': 3,
                        'accepted': 2,
                        'rejected': 9,
                        'artifact': 5
                        }

        with open(self.file_names['cluster_metrics'], 'r') as f:
            metrics = json.load(f)['clusters']
            # loads a list of dictionaries with keys ['label', 'metrics']

        with h5py.File(self.file_names['kwik'], 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group(
                '{}'.format(self.chan_group))
            clusters_group = chan_group.require_group('clusters')
            desc_group = clusters_group.require_group(name)

            for metric in metrics:
                #clu_type = [x[1] for x in self.grp if x[0] == metric['label']]
                #module_logger.info('metrics {}'.format(metric['metrics']))
                # if there are tags in the json file, use them
                # otherwise default is 'unsorted'
                try:
                    clu_type = metric['tags'][0]
                except IndexError:
                    clu_type = 'unsorted'
                    metric['tags'] = ['unsorted']

                attribs = [{'name': 'cluster_group',
                            'data': clu_grp_dict[clu_type],
                            'dtype': np.int64}]

                this_cluster_group = insert_group(
                    desc_group, str(metric['label']), attribs)
                h5t.append_atrributes(this_cluster_group, metric['metrics'])
                tags_list_utf8 = [h5t.h5_unicode_hack(
                    x) for x in metric['tags']]
                h5t.append_atrributes(this_cluster_group,
                                      {'tags': tags_list_utf8})


class KiloKwikWriter(KwikFileWriter):
    mda_params = dict()
    # init the KwikFile class

    def __init__(self, file_names, chan_group=0, rec_in_binary=np.empty(0), raw_format='kwd'):
        super(KiloKwikWriter, self).__init__(file_names, chan_group=chan_group,
                                             rec_in_binary=rec_in_binary,
                                             raw_format=raw_format)

    def load_grp_file(self):
        # look for the first ['grp'] file that exists
        # fist check the 'grp' (user defined), otherwise try the 'grp_kilo' (cluster_KSLabel)
        if os.path.exists(self.file_names['grp']):
            grp_file_path = self.file_names['grp']
        else:
            grp_file_path = self.file_names['grp_kilo']

        #module_logger.info('look for cluster tags file in {}'.format(grp_file_path))
        if os.path.exists(grp_file_path):
            module_logger.info('found cluster tags file in {}'.format(grp_file_path))
            return np.loadtxt(grp_file_path,
                              dtype={'names': ('cluster_id', 'group'),
                                     'formats': ('i2', 'S8')},
                              skiprows=1)
        else:
            module_logger.info('No cluster file tags found. Assume all clu are unsorted')
            return np.empty(0)

    def get_clusters(self):
        # get the clusters from the spike_clusters.npy (if there was manual sorting)
        # or the spike_templates.npy (if there wasn't)
        if self.file_names['clu']:
            self.clu = np.squeeze(np.load(self.file_names['clu']))
        elif self.file_names['temp']:
            self.clu = np.squeeze(np.load(self.file_names['temp']))
        else:
            raise IOError(
                'both spike_clusters.npy and spike_templates.npy weren\'t found')

        # get the spike times from the spike_times.npy file
        self.spk = np.load(self.file_names['spk'])

        # all clusters are 'unsorted' by default.
        # they get their label (if any) in self.make_clu_groups()
        self.grp = [(i, 'unsorted') for i in np.unique(self.clu)]

    def make_clu_groups(self, name='main'):
        clu_grp_dict = {'mua': 1,
                        'noise': 0,
                        'unsorted': 3,
                        'accepted': 2,
                        'rejected': 9,
                        'artifact': 5
                        }
        clu_translate = {'noise': 'noise',
                         'good': 'accepted',
                         'mua': 'mua'}
        # by default, everything is unsorted.
        # if there is a description 'grp' file, look the described and change their label
        sorted_grp = self.load_grp_file()
        # the pc_feature npy files only make sense if there were no clusters created/merged
        # during manual sorting; 
        # in which case the last clu in ind is the last clu in the cluster_group
        auto_cul_grp = np.loadtxt(self.file_names['grp_kilo'],
                              dtype={'names': ('cluster_id', 'group'),
                                     'formats': ('i2', 'S8')},
                              skiprows=1)
        module_logger.debug('max clu {}'.format(np.max(self.clu)))
        if np.max(self.clu) == np.max(auto_cul_grp['cluster_id']):
            pc_ind = np.load(self.file_names['pc_ind'])
        else:
            warnings.warn('Clusters in spike_clusters.npy are not the same as  \
                         in cluster_group.tsv. Features, templates and amplitues will be wrong\
                             or absent')
            pc_ind = np.ones_like(self.clu)
            pc_ind[:] = np.nan
 
        with h5py.File(self.file_names['kwik'], 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group(
                '{}'.format(self.chan_group))
            clusters_group = chan_group.require_group('clusters')
            desc_group = clusters_group.require_group(name)

            for clu in self.grp:
                #clu_type = [x[1] for x in self.grp if x[0] == metric['label']]
                #module_logger.info('metrics {}'.format(metric['metrics']))
                # if there are tags in the json file, use them
                # otherwise default is 'unsorted'
                clu_id, clu_type = clu
                try:
                    clu_type = sorted_grp[sorted_grp['cluster_id'] == clu_id]['group'].astype(str)[
                        0]
                    clu_type = clu_translate[clu_type]
                except IndexError:
                    pass

                attribs = [{'name': 'cluster_group',
                            'data': clu_grp_dict[clu_type],
                            'dtype': np.int64}]

                this_cluster_group = insert_group(desc_group,
                                                  str(clu_id), attribs)
                #h5t.append_atrributes(this_cluster_group, metric['metrics'])
                tags_list_utf8 = [h5t.h5_unicode_hack(clu_type)]
                h5t.append_atrributes(this_cluster_group,
                                      {'tags': tags_list_utf8,
                                       'main_chan': pc_ind[clu_id]})


def make_shank_kwd(raw_file, out_file_path, chan_list):
    raise NotImplementedError
    # ss_file = h5py.File(out_file_path, 'w')
    # h5t.copy_attribs(raw_file, ss_file)
    # ss_file.create_group('/recordings')
    # create_data_groups(raw_file, ss_file, chan_list)
    # ss_file.close()


def load_grp_file(grp_file_path):
    if os.path.exists(grp_file_path):
        return np.loadtxt(grp_file_path,
                          dtype={'names': ('cluster_id', 'group'),
                                 'formats': ('i2', 'S8')},
                          skiprows=1)
    else:
        return None


# offset the recs
@jit
def ref_to_rec_starts(rec_sizes: dict, t_array: np.ndarray) -> (np.ndarray, np.ndarray):
    start = 0
    t_rec = np.empty_like(t_array)
    rec_array = np.empty_like(t_array)

    for rec, size in rec_sizes.items():
        end = start + size
        this_rec_spk = (t_array >= start) & (t_array < end)
        t_rec[this_rec_spk] = t_array[this_rec_spk] - start
        rec_array[this_rec_spk] = rec
        start = end

    return rec_array, t_rec


def mda_to_kwik(kwd_path: str, kwik_path: str, mda_firings_path: str, metrics_path: str,
                realign_to_recordings: bool = True,
                rec_in_binary: np.ndarray = np.empty(0)):
    file_names = {'kwd': kwd_path,
                  'kwik': kwik_path,
                  'firings': mda_firings_path,
                  'cluster_metrics': metrics_path}

    module_logger.info('Creating kwik file {} from firings {}'.format(
        kwik_path, mda_firings_path))
    os.makedirs(os.path.split(kwik_path)[0], exist_ok=True)
    kwik_file_writer = MdaKwikWriter(file_names, rec_in_binary=rec_in_binary)
    kwik_file_writer.get_clusters()
    kwik_file_writer.make_spk_tables(realign_to_recordings=True)
    kwik_file_writer.make_rec_groups()
    kwik_file_writer.make_clu_groups()
    return kwik_file_writer


def kilo_to_kwik(kwd_path: str, kwik_path: str, kilo_folder: str,
                 realign_to_recordings: bool = True,
                 rec_in_binary: np.ndarray = np.empty(0),
                 chan_group=0,
                 kilo_file_names=None,
                 raw_format='kwd'):

    module_logger.info('Creating kwik file {} from kilosort folder {}'.format(
        kwik_path, kilo_folder))
    os.makedirs(os.path.split(kwik_path)[0], exist_ok=True)
    
    if kilo_file_names is None:
        # see explanations of kilo output files in
        # https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md
        kilo_file_names = dict(
            clu='spike_clusters.npy',
            spk='spike_times.npy',
            grp='cluster_group.tsv', # manually curated cluster tags
            grp_kilo='cluster_KSLabel.tsv', # default kilo cluster tags
            phy_par='params.py',
            temp='spike_templates.npy',
            params='params.json',
            pc_ind='pc_feature_ind.npy'
            )

    file_paths = {k: os.path.join(kilo_folder, v)
                  for k, v in kilo_file_names.items()}

    file_paths.update(dict(kwd=kwd_path,kwik=kwik_path))

    # Check whether there is manual sort or not:
    if not os.path.isfile(file_paths['clu']):
        module_logger.info(
            'Clu not found, will assume no manual sorting was done. It may fail')
        file_paths['clu'] = None
        file_paths['grp'] = None
        module_logger.debug(file_paths)
    else:
        module_logger.info(
            'Found clu file, will attempt to unpack manual sorted data from kilosort')
        file_paths['temp'] = None
        module_logger.debug(file_paths)

    k = KiloKwikWriter(file_paths, rec_in_binary=rec_in_binary, chan_group=chan_group, 
                       raw_format=raw_format)
    k.get_clusters()
    module_logger.info('Making spike tables')
    k.make_spk_tables(realign_to_recordings=True)
    module_logger.info('Making rec tables (make_rec_groups)')
    k.make_rec_groups()
    module_logger.info('Making cluster group tables')
    k.make_clu_groups()

    # module_logger.info('Moving files to their sort folder')
    # sort_kilo_dir = os.path.join(fn['folders'][location],
    #                              'kilo_{:02d}'.format(chan_group))
    # et.mkdir_p(sort_kilo_dir)
    # py_files = glob.glob(os.path.join(fn['folders'][location], '*.py'))
    # npy_files = glob.glob(os.path.join(fn['folders'][location], '*.npy'))
    # for src in py_files + npy_files:
    #     shutil.move(src, sort_kilo_dir)
    # module_logger.info('Removing temporary .dat file')
    # dat_file = os.path.join(fn['folders'][location], 'experiment.dat')
    # os.remove(dat_file)
    module_logger.info('Done')
