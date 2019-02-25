import socket
import os
import glob
import json
import numpy as np
import pandas as pd
import logging

# pipeline imports
from pipefinch.pipeline import filestructure as et
from pipefinch.pipeline import probes
from pipefinch.neural.convert import intan
from pipefinch.neural.sort.mountain import core as msc
from pipefinch.h5tools.kwik import kutil
from pipefinch.h5tools.kwik import kwdfunctions as kwdf
from pipefinch.h5tools.kwik import kwikfunctions as kwkf
from pipefinch.neural import units

from intan2kwik import kwd

logger = logging.getLogger('pipefinch.pipeline.sort')


def rhd_to_sort(sess_par: dict):
    # get experiment structure and paths
    exp_struct = et.get_exp_struct(sess_par['bird'], sess_par['sess'])
    logger.info('Raw folder {}'.format(exp_struct['folders']['raw']))
    
    kwik_folder = exp_struct['folders']['kwik']
    msort_folder = exp_struct['folders']['msort']
    raw_folder = exp_struct['folders']['raw']
    kwd_path = exp_struct['files']['kwd']
    bin_path = exp_struct['files']['mda_raw']
    
    file_paths, out_folder = msc.make_paths(exp_struct['folders']['msort'])

    # update the kwd file if something to update, create one from rhd files if non-existent
    try:
        _, nu_pd, _ = kwdf.update_kwd(kwd_path, raw_folder)
    except FileNotFoundError:
        os.makedirs(kwik_folder, exist_ok=True)
        first_intan_hdr, sess_pd = kwd.intan_to_kwd(raw_folder, kwd_path)

    # get the whole rec list and the metadata of the session
    all_rec_list = kutil.get_rec_list(kwd_path)
    pd_meta = kwdf.get_all_rec_meta(kwd_path)

    # get the rig parametrs
    rig_par_file = exp_struct['files']['rig']
    with open(rig_par_file, 'r') as fp:
        rig_par = json.load(fp)

    # make the probe file
    # todo: work with multiple probes
    probe_port = rig_par['chan']['port']['probe_0'].strip('-')
    wanted_chans = np.array([probe_port])  # all ephys channels
    chan_list = kwdf.get_all_chan_names(pd_meta, chan_filt=wanted_chans)

    rec_chans = pd_meta.loc[pd_meta['name'] == 0, 'channel_names'].values
    rec_chans_idx = kwdf.find_chan_names_idx(rec_chans[0], chan_list)

    probe = rig_par['probe']['probe_0']['model']
    headstage = rig_par['probe']['probe_0']['headstage']
    probe_chans = rec_chans_idx - np.min(rec_chans_idx)
    probe_geom = probes.make_map(probe, probe_chans)
    np.savetxt(file_paths['geom'], probe_geom, delimiter=',')

    bin_path = exp_struct['files']['mda_raw']
    os.makedirs(exp_struct['folders']['msort'], exist_ok=True)
    bin_file = kwdf.kwd_to_binary(exp_struct['files']['kwd'],
                                  exp_struct['files']['mda_raw'],
                                  chan_list=chan_list,
                                  rec_list=all_rec_list, header='mda')
    all_rec_meta_pd = kwdf.get_all_rec_meta(exp_struct['files']['kwd'])

    
    os.makedirs(exp_struct['folders']['msort'], exist_ok=True)

    # parameters to pass to the msort scripts
    ds_params = {'samplerate': int(kwdf.get_sampling_rate(all_rec_meta_pd, all_rec_list[0])),
                 'detect_sign': -1}
    with open(file_paths['params'], 'w') as fp:
        json.dump(ds_params, fp)
    logger.info('Created session par files {}'.format(file_paths['params']))

    # Sort
    output_dir = os.path.join(exp_struct['folders']['msort'], 'sort_out')
    msc.sort_dataset(file_paths=file_paths, adjacency_radius=-
                     1, detect_threshold=2, freq_min=600)

    logger.info('Finished sorting, output in {]'.format(output_dir))
    logger.info('To curate, go to msort directory and run qt-mountainview \
        --raw raw.mda --filt sort_out/filt.mda.prv --pre sort_out/pre.mda.prv --samplerate=20000 \
            --firings sort_out/firings.mda --cluster_metrics sort_out/cluster_metrics.json')
    return


def sort_to_kwik(sess_par: dict, firings_to_save='firings_curated'):
    # firings_to_save = 'firings' # 'curated' or 'firings' for default_output
    exp_struct = et.get_exp_struct(sess_par['bird'], sess_par['sess'])
    file_paths, out_folder = msc.make_paths(exp_struct['folders']['msort'])
    kwd_path = exp_struct['files']['kwd']
    kwik_path = exp_struct['files']['kwik']

    metrics_to_save = 'cluster_metrics_curated' if firings_to_save == 'firings_curated' else 'cluster_metrics'
    # make the kwik file
    kwkf.mda_to_kwik(kwd_path,
                     kwik_path,
                     file_paths[firings_to_save],
                     file_paths[metrics_to_save])
    # extract all unit waveforms
    units.get_all_unit_waveforms(kwik_path, kwd_path)
