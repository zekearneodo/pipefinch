# functions to read SpikeGLX neuropixels phase 3B (imec system) file structure, metadata, and data
import os
import glob
import logging
import pandas as pd

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

