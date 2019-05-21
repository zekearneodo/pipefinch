import logging
import os
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import traceback

from pipefinch.h5tools.kwik import kutil
from pipefinch.h5tools.kwik import kwdfunctions as kwdf
from pipefinch.h5tools.kwik import kwikfunctions as kwkf
from pipefinch.h5tools.kwik import event as ev
from pipefinch.pipeline import filestructure as et
from pipefinch.pipeline import rigutil

from pipefinch.neural import units as un
from pipefinch.neural import basic_plot as bp

from pipefinch.util import spectral as sp

logger = logging.getLogger('pipefinch.neural.postsort.session')

# A session contains all the metadata of the session, pointers to the relevant files,
# un and events to plot event-aligned rasters and clusters info


class Session:
    exp_struct = dict()  # experiment file structure
    sess_par = dict()  # session parameters (bird, rec session)
    viz_par = dict()  # visualization parameters
    evt_par = dict()  # event parameters
    rig_par = dict()  # here goes the .json file describing the rig (what channels where what)

    unit = tuple()  # tuple with the unit
    sorted_recs = np.empty(0)
    events = {}
    pd_meta = pd.DataFrame()
    units_meta_pd = pd.DataFrame()

    # quick access
    kwd_path = ''
    kwik_path = ''
    kwe_path = ''
    evt_ttl_pd = pd.DataFrame()

    mic_streams = np.empty(0)

    filters = dict()

    def __init__(self, sess_par, viz_par):
        self.sess_par = sess_par
        self.viz_par = viz_par

        # file structure and locations
        try:
            sort_version = sess_par['sort']
        except:
            sess_par['sort'] = None
        exp_struct = et.get_exp_struct(
            sess_par['bird'], sess_par['sess'], sess_par['sort'])
        self.exp_struct = exp_struct
        self.kwd_path = exp_struct['files']['kwd']
        self.kwik_path = exp_struct['files']['kwik']
        self.kwe_path = exp_struct['files']['kwe']

        # all the metadata in the kwik file (not filtered to the sorted part)
        self.pd_meta = kwdf.get_all_rec_meta(exp_struct['files']['kwd'])
        # the list of recs that are sorted (as per the kwik file)
        self.sorted_recs = kwdf.get_rec_list(exp_struct['files']['kwik'])
        # handy metadata on how the data was recorded
        self.s_f = kutil.get_record_sampling_frequency(
            exp_struct['files']['kwd'])
        self.probe_port = et.get_probe_port(exp_struct, sess_par['probe'])

        # get all the units meta
        self.get_units_meta()

        # load the rig parameters
        self.load_rig_par()

        # set visualization parameters
        self.reset_viz_par()

        self.filters['neural'] = signal.butter(10, 300, 'hp',
                                               fs=self.s_f,
                                               output='sos')

    def reset_viz_par(self):
        vp = self.viz_par
        vp['pre_samples'] = int(vp['pre_ms'] * self.s_f * 0.001)
        vp['post_samples'] = int(vp['post_ms'] * self.s_f * 0.001)
        vp['span'] = vp['post_samples'] - vp['pre_samples']

    def load_rig_par(self, rig_par_dict={}):
        if rig_par_dict:
            self.rig_par = rig_par_dict
        else:
            self.rig_par = et.get_rig_par(self.exp_struct)
            # need to load it from the file.

    def get_sorted_recs(self):
        return self.sorted_recs

    def load_event_ttl(self, ev_name, ch_name):
        ttl = ev.TTL(ev_name, ch_name, self.kwd_path)
        ttl.event_pd.set_index(['ch', 'edge'], inplace=True)
        self.events[ev_name] = ttl

    def get_event_ttl(self):  # get the event in viz_par
        evt_edge = self.viz_par['evt_edge']
        evt_name = self.viz_par['evt_name']
        evt_signal = self.viz_par['evt_signal']
        ch_type, ch_name = rigutil.lookup_signal(self.rig_par, evt_signal)

        try:
            evt_obj = self.events[evt_name]
        except KeyError:
            _, ch_name = rigutil.lookup_signal(self.rig_par, evt_signal)
            self.load_event_ttl(evt_name, ch_name)
        self.evt_ttl_pd = self.events[evt_name].event_pd.loc[ch_name].loc[evt_edge]

    def get_event_stamps(self, filter_recs=np.empty(0)):
        self.get_event_ttl()
        evt_ttl_pd = self.evt_ttl_pd

        if filter_recs.size > 0:
            logger.debug('filter recs')
            evt_ttl_pd = evt_ttl_pd[evt_ttl_pd['rec'].apply(
                lambda x: x in filter_recs)]

        all_rec = evt_ttl_pd['rec'].values
        all_start = evt_ttl_pd['t'].values
        return all_rec, all_start

    def load_mic_peri_event_streams(self, mic='microphone_0'):
        self.mic_streams = self.get_perievent_stream_frames(mic)

    def get_mic_stream(self):
        return self.mic_streams

    def get_perievent_stream_frames(self, signal_name, filter_recs=np.empty(0)) -> np.ndarray:
        ch_type, ch_name = rigutil.lookup_signal(self.rig_par, signal_name)
        wanted_chans = np.array([ch_name])
        sel_chan_names = kwdf.get_all_chan_names(
            self.pd_meta, chan_filt=wanted_chans)

        all_rec, all_start = self.get_event_stamps(filter_recs=filter_recs)

        vp = self.viz_par
        stream_pst_array = kwdf.get_frames(self.kwd_path,
                                           all_start + vp['pre_samples'],
                                           all_rec, vp['span'],
                                           sel_chan_names,
                                           self.pd_meta)
        # that is a [n_evt, n_samples, n_ch] array
        return stream_pst_array

    def get_perievent_neural_traces(self, unit: un.Unit):
        # get all the ephys channels for the probe around the event
        all_ch_array = self.get_perievent_stream_frames(unit.neural_port,
                                                        filter_recs=self.sorted_recs)
        # filter the main channels of the unit
        # get the unit main channels
        unit_main_chans = unit.get_unit_main_chans()[1]
        #logger.info('unit main chans {}'.format(unit_main_chans))
        return all_ch_array[:, :, unit_main_chans]

    def get_unit(self, clu: int, group: int = 0) -> un.Unit:
        # create a unit object with the kwd, kwik files and a clu number
        return un.Unit(clu, self.kwik_path, self.kwik_path,
                       group=group,
                       port=self.probe_port)

    def get_units_meta(self):
        # - get the recarray of the units in the kwik file
        #   (with the group, clu, qlt)
        # - make it into a pd dataframe
        # - get the tags, main_chans
        all_clu_meta_pd = pd.DataFrame(kwkf.list_units(self.kwik_path))

        # get extra features
        d = {'tags': []}
        for clu in all_clu_meta_pd['clu'].values:
            a_unit = un.Unit(clu, self.kwik_path)
            d['tags'].append(a_unit.get_attrs()['tags'])

        self.units_meta_pd = pd.concat([all_clu_meta_pd, pd.DataFrame(d)],
                                       axis=1)

        for tag in self.get_unit_tags():
            self.units_meta_pd[tag] = self.units_meta_pd['tags'].apply(
                lambda x: tag in x)

    def get_unit_tags(self):
        if self.units_meta_pd.size == 0:
            self.get_units_meta()
        return np.unique(np.concatenate(self.units_meta_pd['tags'].values).flatten())


# functions for viewing object of this session

def plot_unit(sess: Session, clu: int, fig=None, example_event_id: int = 0):

    a_unit = un.Unit(clu, sess.kwik_path, sess.kwd_path, port=sess.probe_port)
    sorted_recs = sess.get_sorted_recs()

    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        fig_ax = {}
    fig_width = 0.7
    # time aligned column
    fig_ax['raster'] = fig.add_axes([0, .75, fig_width, 0.2])
    fig_ax['histogram'] = fig.add_axes([0, .55, fig_width, 0.2])
    fig_ax['neural'] = fig.add_axes([0, .3,
                                     fig_width, 0.25])
    fig_ax['spectrogram'] = fig.add_axes([0, .1,
                                          fig_width, 0.2])
    fig_ax['soundwave'] = fig.add_axes([0, 0,
                                        fig_width, 0.1])

    # unit features column
    fig_ax['waveform'] = fig.add_axes([fig_width + .01, .45 + .01,
                                       1 - fig_width, 0.4 - 0.01])
    fig_ax['correlogram'] = fig.add_axes([fig_width + .01, 0,
                                          1 - fig_width, 0.45 - .12])
    for ax in fig_ax.values():
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    # plot raster, histogram, isi distribution
    all_rec, all_start = sess.get_event_stamps(filter_recs=sorted_recs)
    a_raster = a_unit.get_raster(all_start, all_rec,
                                 sess.viz_par['span'],
                                 return_ms=True)
    bp.plot_trial_raster(a_raster, ev_marks={
                         'song': -sess.viz_par['pre_ms']},
                         ax=fig_ax['raster'])
    bp.plot_trial_raster(a_raster, ev_marks={
                         'song': -sess.viz_par['pre_ms']},
                         bin_size=15,
                         ax=fig_ax['histogram'])
    #fig_ax['histogram'].yaxis.set_ticks([0, a_raster.shape[0]])
    #fig_ax['raster'].yaxis.set_ticks([0, a_raster.shape[0]])

    # instad of spectrogram, plot one example song
    ax = fig_ax['neural']
    neural_chan_filter = sess.filters['neural']
    neural_streams = sess.get_perievent_neural_traces(a_unit)[
        example_event_id, :, :]
    neural_filtered = signal.sosfiltfilt(neural_chan_filter,
                                         neural_streams, axis=0)
    arr_plot = bp.plottable_array(neural_filtered,
                                  1/np.ptp(neural_filtered, axis=0),
                                  np.arange(neural_filtered.shape[0]))
    t_ms = np.arange(neural_filtered.shape[0])/sess.s_f * 1000
    ax.plot(t_ms, neural_filtered)
    #ax.xaxis.set_ticks(np.arange(100, t_ms[-1], 500))
    ax.axvline(x=-sess.viz_par['pre_ms'])
    ax.set_ylabel('ch')
    ax.set_xlim(0, np.max(t_ms))

    # get the mic stream and set the timescale for the streams
    mic_stream = sess.get_mic_stream()[example_event_id]
    t_ms = np.arange(mic_stream.size)/sess.s_f * 1000

    # plot one example spectrogram
    ax = fig_ax['spectrogram']
    f_spec, t_spec, spec = sp.ms_spectrogram(mic_stream.flatten(), sess.s_f)
    ax.pcolormesh(t_spec*1000, f_spec, np.log(spec),
                  rasterized=True, cmap='inferno')
    ax.axvline(x=-sess.viz_par['pre_ms'])
    ax.set_xlim(0, np.max(t_ms))

    # plot one example song
    ax = fig_ax['soundwave']
    mic_stream = sess.get_mic_stream()[example_event_id]
    t_ms = np.arange(mic_stream.size)/sess.s_f * 1000
    ax.plot(t_ms, mic_stream, 'y')
    ax.xaxis.set_ticks(np.arange(100, t_ms[-1], 500))
    ax.axvline(x=-sess.viz_par['pre_ms'])
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('mic')
    ax.set_xlim(0, np.max(t_ms))

    # pot isi, rec presence
    ax = fig_ax['correlogram']
    isi_dist = a_unit.get_isi_dist(max_t=25)
    isi_total = np.sum(isi_dist[1])
    isi_mua_threshold = int(isi_total*0.01)

    ax.bar(*isi_dist, linewidth=0, color='C9')

    ax.set_xlabel('ISI (ms)')
    ax.xaxis.set_ticks([0, 10])
    ax.yaxis.set_ticks([isi_mua_threshold])
    ax.yaxis.tick_right()
    ax.axhline(y=isi_mua_threshold, color='C7', linestyle=':')
    ax.axvline(x=1, color='C5', linestyle=':')

    # plot main waveform
    main_wave = a_unit.get_unit_main_wave(n_chans=4)
    main_chan_names = a_unit.get_unit_main_chans_names(n_chans=4)
    ax = fig_ax['waveform']
    # ax.plot(main_wave.mean(axis=0))
    #sns.tsplot(main_wave, ax=ax, err_style='boot_traces', n_boot=100)
    #sns.tsplot(main_wave, ax=ax, ci=95, )
    ax.plot(np.mean(main_wave, axis=0))
    ax.legend(main_chan_names, ncol=2, fontsize='small',
              loc='upper right', bbox_to_anchor=(1., 1.4))
    ax.xaxis.set_ticks([0, 20])
    ax.set_xlabel('t (samples)')
    ax.set_ylabel('main spk waveform')
    ax.yaxis.set_label_position("right")

    fig.suptitle('Clu {}, {}'.format(clu, a_unit.get_attrs()['tags']))
    # plt.tight_layout()
    return fig


def plot_all_units(sess, only_tags=['accepted', 'mua', 'unsorted'], example_event_idx: int = 0):
    units_list = kwkf.list_units(sess.kwik_path)
    kwik_folder = os.path.split(sess.kwik_path)[0]
    rasters_path = os.path.join(kwik_folder, 'rasters')
    os.makedirs(rasters_path, exist_ok=True)
    for clu in units_list['clu']:
        a_unit = un.Unit(clu, sess.kwik_path, sess.kwd_path,
                         port=sess.probe_port)
        unit_tags = a_unit.get_attrs()['tags']
        # if it is a rejected one don't even care
        sort_ver_string = ''
        try:
            sort_ver_string = '{}'.format(sess_par['sort'])
        except:
            sort_ver_string = 'n'

        if (not 'rejected' in unit_tags) and any([x in unit_tags for x in only_tags]):
            try:
                fig = plot_unit(sess, clu, example_event_id=example_event_idx)
                fig_file = '{}_unit_{}_{:03d}.png'.format(unit_tags[0], sort_ver_string, clu)
                fig.savefig(os.path.join(rasters_path, fig_file),
                            bbox_inches='tight')
            except Exception as err:
                logger.warn('Failed to plot clu {}, error {}'.format(clu, err))
                traceback.print_exc()
                continue
