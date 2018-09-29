import os
import logging
import json
import warnings

from mountainlab_pytools import mdaio
from mountainlab_pytools import mlproc as mlp

logger = logging.getLogger('pipefinch.sort.mountain.comre')


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

    input_f_names = {'mda': 'raw.mda',
                     'params': 'params.json',
                     'geom': 'geom.csv'}

    output_f_names = {'filt': 'filt.mda.prv',
                      'pre': 'pre.mda.prv',
                      'firings': 'firings.mda',
                      'firings_curated': 'firings_curated.mda',
                      'cluster_metrics': 'cluster_metrics.json'
                      }
    out_folder_path = os.path.join(ss_folder, out_subfolder_name)
    file_paths = {k: os.path.join(ss_folder, v)
                  for k, v in input_f_names.items()}
    file_paths.update({k: os.path.join(ss_folder, out_subfolder_name, v)
                       for k, v in output_f_names.items()})
    return file_paths, out_folder_path


def sort_dataset(*, file_paths: dict, freq_min: int=300, freq_max: int=6000,
                 adjacency_radius: int=1, detect_threshold: float=3, opts: dict={}):
    dataset_dir = os.path.split(file_paths['mda'])[0]
    output_dir = os.path.split(file_paths['filt'])[0]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Dataset parameters
    ds_params = read_dataset_params(dataset_dir)

    # Bandpass filter
    logger.info('Bandpass filter')
    bandpass_filter(
        timeseries=file_paths['mda'],
        timeseries_out=file_paths['filt'],
        samplerate=ds_params['samplerate'],
        freq_min=freq_min,
        freq_max=freq_max,
        opts=opts
    )

    # Whiten
    logger.info('Whitening')
    whiten(
        timeseries=file_paths['filt'],
        timeseries_out=file_paths['pre'],
        opts=opts
    )

    # Sort
    detect_sign = 1
    if 'spike_sign' in ds_params:
        detect_sign = ds_params['spike_sign']
    if 'detect_sign' in ds_params:
        detect_sign = ds_params['detect_sign']
    #
    geom = file_paths['geom'] if os.path.isfile(file_paths['geom']) else None

    logger.info('Sorting')
    ms4alg_sort(
        timeseries=file_paths['pre'],
        geom=geom,
        firings_out=file_paths['firings'],
        adjacency_radius=adjacency_radius,
        detect_sign=detect_sign,
        detect_threshold=detect_threshold,
        opts=opts
    )

    # Compute cluster metrics
    logger.info('Getting cluster metrics')
    compute_cluster_metrics(
        timeseries=file_paths['pre'],
        firings=file_paths['firings'],
        metrics_out=file_paths['cluster_metrics'],
        samplerate=ds_params['samplerate'],
        opts=opts
    )

    # Automated curation
    logger.info('Automatically curating')
    automated_curation(
        firings=file_paths['firings'],
        cluster_metrics=file_paths['cluster_metrics'],
        firings_out=file_paths['firings_curated'],
        opts=opts
    )


# THE FUNCTIONS THAT RUN PROCESSES OR ADD THEM TO A Pipeline object

def read_dataset_params(dsdir: str) -> dict:
    params_fname = mlp.realizeFile(dsdir+'/params.json')
    if not os.path.exists(params_fname):
        raise Exception(
            'Dataset parameter file does not exist: ' + params_fname)
    with open(params_fname) as f:
        return json.load(f)


def bandpass_filter(*, timeseries, timeseries_out, samplerate, freq_min, freq_max, opts={}):
    return mlp.runProcess('ephys.bandpass_filter',
                          {'timeseries': timeseries},
                          {'timeseries_out': timeseries_out},
                          {'samplerate': samplerate,
                              'freq_min': freq_min,
                              'freq_max': freq_max},
                          opts)


def whiten(*, timeseries, timeseries_out, opts={}):
    return mlp.runProcess('ephys.whiten',
                          {'timeseries': timeseries},
                          {'timeseries_out': timeseries_out},
                          {},
                          opts)


def ms4alg_sort(*, timeseries, geom, firings_out, detect_sign, adjacency_radius, detect_threshold=3, opts={}):
    pp = {}
    pp['detect_sign'] = detect_sign
    pp['adjacency_radius'] = adjacency_radius
    pp['detect_threshold'] = detect_threshold

    input_dict = {'timeseries': timeseries}
    if geom is not None:
        input_dict['geom'] = geom
    else:
        warnings.warn('Will sort with no geometry input')

    mlp.runProcess('ms4alg.sort',
                   input_dict,
                   {'firings_out': firings_out}, pp,
                   opts)


def compute_cluster_metrics(*, timeseries, firings, metrics_out, samplerate, opts={}):

    metrics_clu = mlp.runProcess('ms3.cluster_metrics',
                                 {'timeseries': timeseries, 'firings': firings},
                                 {'cluster_metrics_out': True},
                                 {'samplerate': samplerate},
                                 opts)['cluster_metrics_out']

    metrics_iso = mlp.runProcess('ms3.isolation_metrics',
                                 {'timeseries': timeseries, 'firings': firings},
                                 {'metrics_out': True},
                                 {'compute_bursting_parents': 'true'},
                                 opts)['metrics_out']

    return mlp.runProcess('ms3.combine_cluster_metrics',
                          {'metrics_list': [metrics_clu, metrics_iso]},
                          {'metrics_out': metrics_out},
                          {},
                          opts)


def automated_curation(*, firings, cluster_metrics, firings_out, opts={}):
    # Automated curation
    label_map = mlp.runProcess('ms4alg.create_label_map',
                               {'metrics': cluster_metrics},
                               {'label_map_out': True},
                               {},
                               opts)['label_map_out']

    return mlp.runProcess('ms4alg.apply_label_map',
                          {'label_map': label_map, 'firings': firings},
                          {'firings_out': firings_out},
                          {},
                          opts)
