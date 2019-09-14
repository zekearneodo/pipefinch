# objects to do quick stuff with clusters
from __future__ import division

import numpy as np
import logging
import os
import pickle
import h5py
import json

from tqdm import tqdm_notebook as tqdm
from numba import jit
from joblib import Parallel, delayed

from pipefinch.h5tools.kwik import kwdfunctions as kwdf
from pipefinch.h5tools.kwik import kwikfunctions as kwkf
from pipefinch.h5tools.core import h5tools as h5t

from pipefinch.util import jitarray as jita
from pipefinch.pipeline import sglxutil as sglu

logger = logging.getLogger('pipefinch.neural.units')
# from importlib import reload
# reload(kwdf)


@jit(nopython=True)
def offset_timestamps(t, r, rec_offset):
    t_offset = np.zeros_like(t)
    for t_i, r_i in zip(t, r):
        t_i += rec_offset[r_i]
    return t_offset


class Unit:
    def __init__(self, clu: int, kwik_path: str, kwd_path: str = '', group=0, port='A'):

        self.clu = clu
        self.group = group
        self.bird = None
        self.sess = None

        self.kwik_path = kwik_path
        self.kwd_path = kwd_path

        self.type = 'cluster'
        self.id = None
        self.metrics = None

        self.attrs = None
        self.qlt = None
        self.time_samples = None
        self.recordings = None
        self.recording_offsets = None
        self.sampling_rate = None

        self.kwd_file = None
        self.all_waveforms = None  # all of the waveforms
        self.n_waveforms = 10000  # sample of waveforms to show/compute
        self.waveforms = None
        self.avg_waveform = None
        self.main_chan = None
        self.main_wave = None
        self.waveform_pars = {}
        self.unit_chan_names = np.empty(0)
        self.unit_chans = np.empty(0)

        self.neural_port = port

        self.get_attrs()
        self.get_sampling_rate()
        # self.get_qlt()
        if not kwd_path == '':
            self.get_rec_offsets()
            self.get_unit_chans()
        self.get_time_stamps()

    # get time stamps of spiking events (in samples)

    def get_time_stamps(self, absolute_time: np.bool = False):
        assert (self.kwik_path is not None)
        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(
            self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)
        r_path = "/channel_groups/{0:d}/spikes/recording".format(self.group)

        with h5py.File(self.kwik_path, 'r') as kwik_file:
            # time samples are relative to the beginning of the corresponding rec
            this_clu = kwik_file[clu_path][:] == self.clu
            all_t = kwik_file[t_path][:]
            all_rec = kwik_file[r_path][:]

            dtype = kwik_file[t_path].dtype
            self.time_samples = np.array(
                all_t[this_clu], dtype=np.dtype(dtype))

            dtype = kwik_file[r_path].dtype
            # recordings ids (as in the key)
            self.recordings = np.array(
                all_rec[this_clu], dtype=np.dtype(dtype))
        return self.time_samples, self.recordings

    def get_attrs(self):
        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(
            self.group, self.clu)
        with h5py.File(self.kwik_path, 'r') as kwik_file:
            self.attrs = h5t.obj_attrs_2_dict_translator(kwik_file[path])
        return self.attrs

    def get_rec_offsets(self):
        if self.kwd_path == "":
            # No kwd path is spikegl; only one recording
            self.recording_offsets = {0: 0}
        else:
            self.recording_offsets = kwdf.get_rec_starts(self.kwd_path)
        return self.recording_offsets

    def get_sampling_rate(self):
        # get the sampling rate from the first rec in the .kwik file.
        # the sampling_rate should be the sam across all recs anyway.
        first_rec = kwdf.get_rec_list(self.kwik_path)[0]

        self.sampling_rate = kwkf.get_rec_attrs(self.kwik_path, 0)[
            'rec_group']['sample_rate']
        return self.sampling_rate

    # get the quality of the unit
    def get_qlt(self):
        assert (self.kwik_path is not None)
        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(
            self.group, self.clu)
        with h5py.File(self.kwik_path, 'r') as kwik_file:
            self.qlt = self.kwik_path[path].attrs.get('cluster_group')

    def get_sess_par(self):
        pass
        # folder = self.get_folder()
        # bird_folder, sess = os.path.split(folder)
        # self.bird = str(os.path.split(bird_folder)[1])
        # self.sess = str(sess)
        # self.id = 'unit_{}_{}_{}_{}'.format(self.bird, self.sess, self.group, self.clu)

    def get_raster(self, starts: np.array, recs: np.array, span: int, return_ms=False) -> np.array:
        """generates a compact raster in timestamp units around starts

        Arguments:
            starts {np.array} -- array of starting time stamps
            recs {np.array} -- array of rec to which start time stamps belong
            span {int} -- window to gather events

        Returns:
            np.array -- (n_trial, span)
        """
        # starts: samples, relative to rec
        # span: samples
        # recs: rec for reference
        # the kwik file has timestamps relative to beginning of each rec
        assert (starts.size == recs.size)
        rec_list = kwdf.get_rec_list(self.kwik_path)
        start_rec_offsets = np.zeros_like(rec_list)

        rows = starts.shape[0]
        cols = span
        if return_ms:
            cols = int(cols * 1000/self.sampling_rate)
        raster = np.empty((rows, cols), dtype=np.float64)
        raster[:] = np.nan

        # do the raster in samples
        i_trial = 0
        t = self.time_samples
        r = self.recordings

        for rec in np.unique(recs):
            rec_time_samples = t[r == rec]
            for trial_start in starts[recs == rec]:
                end = np.int(trial_start + span)
                where = (rec_time_samples[:] >= trial_start) & (
                    rec_time_samples[:] <= end)
                n = np.sum(where)
                trial_t_samples = rec_time_samples[where] - trial_start
                if return_ms:
                    raster[i_trial, :n] = np.round(
                        trial_t_samples * 1000. / self.sampling_rate)
                else:
                    raster[i_trial, :n] = trial_t_samples
                i_trial += 1
        return raster

    def get_isi(self):
        if self.time_samples is None:
            self.get_time_stamps()

        if self.recording_offsets is None:
            self.get_rec_offsets()

        rec_starts = np.array([self.recording_offsets[r]
                               for r in self.recordings])
        offset_timestamps = self.time_samples + rec_starts
        offset_timestamps.sort()

        all_isi_ms = np.round(np.diff(offset_timestamps) /
                              (self.sampling_rate * 0.001))
        return all_isi_ms

    def get_isi_dist(self, bin_size_ms=0.5, max_t=20):

        all_isi_ms = self.get_isi()

        bins = np.arange(0, max_t, bin_size_ms)
        hist, bins = np.histogram(all_isi_ms, bins)

        bins = bins[:-1]
        #two_side_bins = np.concatenate([-bins[::-1], bins[1:]])
        #two_side_hist = np.concatenate([hist[::-1], hist[1:]])
        return bins, hist

    def get_folder(self):
        return os.path.split(os.path.abspath(self.kwik_path))[0]

    def get_unit_chans(self):
        # this is inefficiet and can lead to errors if channels change across recordings
        # the right thing to do is to fill it up when going from mda to kwik
        # load parameters
        # logger.warning(
        #     'You are getting channel locations from one rec and using for all, mind that this only works if all recs have the same setting')
        if self.kwd_path == "":
            logger.info('No kwd file, no unit chans')
        sess_meta_pd = kwdf.get_all_rec_meta(self.kwd_path)
        wanted_chans = np.array([self.neural_port + '-'])
        chan_names = kwdf.get_all_chan_names(
            sess_meta_pd, chan_filt=wanted_chans)
        sess_chans = kwdf.rec_chan_idx(
            sess_meta_pd, 0, chan_names, block='analog')

        self.unit_chan_names = chan_names
        self.unit_chans = sess_chans
        return sess_chans

    def get_unit_chan_names(self):
        if self.unit_chan_names.size == 0:
            self.get_unit_chans()
        return self.unit_chan_names

    def save_unit_spikes(self):
        unit_path = self.get_folder()
        file_folder = os.path.join(unit_path, 'unit_waveforms')
        os.makedirs(file_folder, exist_ok=True)
        file_path = os.path.join(file_folder,
                                 'unit_{}_{:03d}.npy'.format(self.group,
                                                             self.clu))
        logger.info('Saving unit {0} in file {1}'.format(self.clu, file_path))
        np.save(file_path, self.all_waveforms)
        par_path = os.path.join(file_folder,
                                'unit_{}_{:03d}.par.pickle'.format(self.group,
                                                                   self.clu))
        pickle.dump(self.waveform_pars, open(par_path, 'wb'))

    def load_unit_spikes(self):
        logger.debug('will try to load previous unit files')
        # careful, loads the last saved
        folder = self.get_folder()
        f_name = 'unit_{}_{:03d}.npy'.format(self.group, self.clu)
        p_name = 'unit_{}_{:03d}.par.pickle'.format(self.group, self.clu)
        self.waveform_pars = pickle.load(open(os.path.join(folder, 'unit_waveforms', p_name),
                                              'rb'))
        self.all_waveforms = np.load(os.path.join(
            folder, 'unit_waveforms', f_name), mmap_mode='r')

        return self.all_waveforms

    def get_unit_spikes(self, before: int = 20, after: int = 20, max_events=5000) -> np.ndarray:
        # wavefomrs is an array [n_spikes, n_samples, n_channels]
        logger.debug('Getting units for clu {} in file {}'.format(
            self.clu, self.kwik_path))
        valid_times = self.time_samples[self.time_samples > before]
        valid_recs = self.recordings[self.time_samples > before]

        if valid_times.size < self.time_samples.size:
            logger.warn(
                'Some frames were out of left bounds and will be discarded')
            logger.warn(
                'will collect only {0} events...'.format(valid_times.size))

        chan_list = self.get_unit_chans()
        logger.debug('chanlist {}'.format(chan_list))

        self.waveform_pars = {'before': before,
                              'after': after,
                              'chan_list': np.array(chan_list)}

        try:
            assert valid_times.size > 1, 'no valid events'
            # get a random sample of max_events elements
            sample = np.random.choice(np.arange(valid_times.size),
                                      size=min(max_events, valid_times.size),
                                      replace=False)
            self.all_waveforms = kwdf.collect_frames_fast(self.kwd_path,
                                                          valid_recs[sample],
                                                          valid_times[sample] -
                                                          before,
                                                          before + after,
                                                          np.array(chan_list))
        except (ValueError, AssertionError) as err:
            logger.warn(
                'Could not retrieve waveforms for clu {}, error'.format(self.clu, err))
            self.all_waveforms = np.zeros(
                [1, before + after, np.array(chan_list).size])
            self.all_waveforms[:] = np.nan
        self.save_unit_spikes()
        return self.waveforms

    def load_all_waveforms(self):
        folder = self.get_folder()
        f_name = 'unit_{}_{:03d}.npy'.format(self.group, self.clu)
        return np.load(os.path.join(folder, 'unit_waveforms', f_name), mmap_mode='r')

    def set_n_waveforms(self, n_waveforms):
        self.n_waveforms = n_waveforms

    def get_waveforms(self, before=20, after=20, force=False):
        try:
            logger.debug('Trying to load waveforms file')
            assert force is False
            self.load_unit_spikes()
        except:
            logger.debug('Could not load, wil try to gather wavefomrs')
            self.get_unit_spikes(before=before, after=after)
            logger.debug('will save the spikes for the nest time around')
            self.save_unit_spikes()
        # all waveforms were loaded into self.all_waveforms.
        # now we want to make a sample fo them in self.waveforms, to show and compute metrics
        self.n_waveforms = min(self.n_waveforms, self.all_waveforms.shape[0])
        waveform_samples = np.random.choice(self.all_waveforms.shape[0], self.n_waveforms,
                                            replace=False)
        self.waveforms = self.all_waveforms[waveform_samples, :, :]
        return self.waveforms

    def get_avg_wave(self, force=False):
        if (self.waveforms is None) or force:
            self.get_waveforms()
        return np.mean(self.waveforms, axis=0)

    def get_unit_main_chan(self):
        a_w_f = self.get_avg_wave()
        main_chan = np.argmax(np.ptp(a_w_f, axis=0))
        main_chan_absolute = self.waveform_pars['chan_list'][main_chan]
        return main_chan, main_chan_absolute

    def get_unit_main_chans(self, n_chans=4):
        a_w_f = self.get_avg_wave()
        main_chans_idx = np.argsort(np.ptp(a_w_f, axis=0))[::-1][:n_chans]
        # logger.info('main chans {}'.format(main_chans))
        main_chan_absolute = np.array(
            self.waveform_pars['chan_list'])[main_chans_idx]
        # main_chan_absolute is the actual order of the channel in the array
        return main_chans_idx.astype(np.int), main_chan_absolute

    def get_unit_main_chans_names(self):
        #main_chans_idx = self.get_unit_main_chans(n_chans=n_chans)
        unit_chans = self.get_unit_chans()
        #logger.info(unit_chans)
        main_chans_idx = np.array([np.where(x == unit_chans)[
                                  0] for x in unit_chans]).squeeze()
        unit_chan_names = self.unit_chan_names
        #logger.info(main_chans_idx)
        return unit_chan_names[main_chans_idx]

    def get_unit_main_wave(self, n_chans=4):
        ch = self.get_unit_main_chans(n_chans=n_chans)[0]
        return self.waveforms[:, :, ch]

    def get_unit_ptp(self):
        wf_main = self.get_unit_main_wave()
        all_ptp = wf_main.ptp(axis=1)
        return np.median(all_ptp), np.std(all_ptp)

    def get_all_unit_widths(self):
        logger.info('Getting width of all spikes from clu {}'.format(self.clu))
        wf_main = self.get_unit_main_wave()
        wf_samples = wf_main.shape[1]
        mid_points = np.min(wf_main, axis=1) + np.ptp(wf_main, axis=1) / 2.
        mid_points_array = np.reshape(
            np.repeat(mid_points, wf_samples), [-1, wf_samples])
        x, y = np.where(np.diff((wf_main > mid_points_array), 1))
        widths = []
        for i in np.unique(x):
            zero_xings = y[x == i]
            if zero_xings.size > 1:
                widths.append(np.max(np.diff(y[x == i])))
        return np.array(widths)

    def get_unit_widths(self):
        widths = self.get_all_unit_widths()
        return np.median(widths), np.std(widths)


class SglUnit(Unit):
    sgl_data_folder = ''
    imec_meta_path = ''
    kilo_folder = ''

    def __init__(self, clu: int, kwik_path: str, sgl_data_path: str = '', group=0, port='A'):
        super().__init__(clu, kwik_path, kwd_path='', group=group, port=port)

    def get_unit_main_chans(self, n_chans=12):
        kilo_main_chans = self.get_attrs()['main_chan']
        assert(n_chans <= kilo_main_chans.size)
        # main_chan_absolute is the actual order of the channel in the array
        return np.argsort(kilo_main_chans[:n_chans]), kilo_main_chans[:n_chans]

    def get_sess_par(self) -> dict:
        sort_folder = self.get_folder()
        path_parts = os.path.normpath(sort_folder).split(os.sep)[::-1]
        sess_pars = ['sort', 'epoch', 'sess']
        sess_par = {attr: val for attr, val in zip(sess_pars, path_parts[:3])}
        sess_par['bird'] = path_parts[-5]
        return sess_par
        # self.bird = str(os.path.split(bird_folder)[1])
        # self.sess = str(sess)
        # self.id = 'unit_{}_{}_{}_{}'.format(self.bird, self.sess, self.group, self.clu)

    def get_exp_struct(self) -> dict:
        sess_par = self.get_sess_par()
        return sglu.sgl_struct(sess_par, sess_par['epoch'])

    def get_unit_chans(self, n_chans=12):
        # overloaded method to just get the unit chans in the binary file.
        # the names are just the number in the binary file
        # n_chans=12, which is the max number kilosort returns

        self.unit_chans = self.get_unit_main_chans(n_chans=n_chans)[1]
        self.unit_chan_names = np.array(
            ['{:03d}'.format(c) for c in self.unit_chans])
        return self.unit_chans
    
    def get_max_amp_chans(self, n_chans=12) -> tuple:
        chan_order, chan = self.get_unit_main_chans(n_chans)
        main_wave = self.get_unit_main_wave(n_chans=n_chans)
        main_chan_names = self.get_unit_main_chans_names()
        
        avg_wav = main_wave.mean(axis=0)
        spk_amps = np.ptp(avg_wav, axis=0)
        sort_ind = np.argsort(spk_amps, axis=0)[::-1]
        
        return sort_ind, chan_order[sort_ind], chan[sort_ind], main_chan_names[sort_ind]
    
    def get_ksort_paths(self):
        sort_folder = self.get_folder()
        ksort_folder = os.path.split(sort_folder)[0]

        sort_paths = {'sort_folder': sort_folder,
                      'sort_bin_file': os.path.join(ksort_folder, 'raw.bin'),
                      'sort_pars_file': os.path.join(ksort_folder, 'params.json')}
        return sort_paths

    def get_sort_pars(self):
        json_pars_path = self.get_exp_struct()['files']['par']
        with open(json_pars_path) as jfile:
            jpar = json.load(jfile)
        return jpar

    def load_bin_file(self):
        pass

    def get_raw_file(self) -> np.ndarray:
        exp_struct = self.get_exp_struct()
        sort_par = self.get_sort_pars()
        raw_ap_path = exp_struct['files']['bin_raw']
        all_raw_arr = np.memmap(raw_ap_path,
                                dtype=sort_par['dtype_name'],
                                mode='r').reshape([-1, sort_par['n_chan']])
        return all_raw_arr

    def get_raw_frames(self, start_samples: np.ndarray, span: int, chans_arr: np.ndarray) -> np.ndarray:
        all_raw_arr = self.get_raw_file()
        frames_arr = jita.collect_frames(all_raw_arr,
                                         start_samples,
                                         span,
                                         chans_arr)
        return frames_arr

    def get_unit_spikes(self, before: int = 20, after: int = 20, max_events=5000) -> np.ndarray:
        # wavefomrs is an array [n_spikes, n_samples, n_channels]
        logger.debug('Getting units for clu {} in file {}'.format(
            self.clu, self.kwik_path))
        valid_times = self.time_samples[self.time_samples > before]
        valid_recs = self.recordings[self.time_samples > before]

        if valid_times.size < self.time_samples.size:
            logger.warn(
                'Some frames were out of left bounds and will be discarded')
            logger.warn(
                'will collect only {0} events...'.format(valid_times.size))

        chan_list = self.get_unit_chans()
        logger.debug('chanlist {}'.format(chan_list))

        self.waveform_pars = {'before': before,
                              'after': after,
                              'chan_list': np.array(chan_list)}

        try:
            assert valid_times.size > 1, 'no valid events'
            # get a random sample of max_events elements
            sample = np.random.choice(np.arange(valid_times.size),
                                      size=min(max_events, valid_times.size),
                                      replace=False)

            start_samples = valid_times[sample] - before
            span = before + after
            self.all_waveforms = self.get_raw_frames(start_samples,
                                                     span,
                                                     np.array(chan_list))

        except (ValueError, AssertionError) as err:
            logger.warn(
                'Could not retrieve waveforms for clu {}, error'.format(self.clu, err))
            self.all_waveforms = np.zeros(
                [1, before + after, np.array(chan_list).size])
            self.all_waveforms[:] = np.nan
        self.save_unit_spikes()
        return self.waveforms


def get_all_unit_waveforms(kwik_path, kwd_path, port='A', before=20, after=20,
                           max_events=5000):
    units_list = kwkf.list_units(kwik_path)
    clu_list = units_list['clu']

    logger.info('About to get all waveforms for {} units in file {}'.format(
        units_list.shape[0], kwik_path))

    # For debugging without the Parallel
    # return Unit(
    #     clu_list[0], kwik_path, kwd_path, port=port).get_unit_spikes(before=before,
    #                                                        after=after,
    #                                                        max_events=max_events)

    def waveform_get(u): return Unit(
        u, kwik_path, kwd_path, port=port).get_unit_spikes(before=before,
                                                           after=after,
                                                           max_events=max_events)

    Parallel(n_jobs=6)(delayed(waveform_get)(clu)
                       for clu in tqdm(clu_list, total=clu_list.size))
    return 0
