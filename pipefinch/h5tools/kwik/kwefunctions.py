# Functions for interacting with the .kwe files (events)
import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rf

from pipefinch.h5tools.core.h5tools import h5_decorator, list_subgroups

# (remember, by default, @h5 decorator will leave a file open, if an open file is entered)

@h5_decorator
def get_events_one_name(kwe_file, ev_type, ev_name, rec=None):
    times_table = kwe_file['event_types'][ev_type][ev_name]['time_samples'][:]
    rec_table = kwe_file['event_types'][ev_type][ev_name]['recording'][:]
    type_table = np.array(['{}'.format(ev_type) for i in range(times_table.size)],
                          dtype='|S32')
    name_table = np.array(['{}'.format(ev_name) for i in range(times_table.size)],
                          dtype='|S32')
    events_recarray = np.rec.fromarrays((times_table,
                                         rec_table,
                                         name_table,
                                         type_table),
                                        dtype=[('t', 'i8'),
                                               ('rec', 'i2'),
                                               ('name', '|S32'),
                                               ('type', '|S32')])
    events_recarray = events_recarray if rec is None else events_recarray[events_recarray.rec == rec]
    return events_recarray

@h5_decorator
def get_events_one_type(kwe_file, ev_type, ev_names=[], rec=None):
    if ev_names == []:
        ev_names = list_events(kwe_file, ev_type)
    ev_stack = [get_events_one_name(kwe_file, ev_type, ev_name, rec=rec) for ev_name in ev_names]
    return rf.stack_arrays(ev_stack, asrecarray=True, usemask=False)

@h5_decorator
def get_all_events(kwe_file, rec=None):
    ev_types = list_event_types(kwe_file)
    ev_stack = [get_events_one_type(kwe_file, ev_type, rec=rec) for ev_type in ev_types]
    return rf.stack_arrays(ev_stack, asrecarray=True, usemask=False)


@h5_decorator
def list_events(kwe_file, ev_type):
    ev_type_group = kwe_file['event_types'][ev_type]
    return list_subgroups(ev_type_group)


@h5_decorator
def list_event_types(kwe_file):
    ev_group = kwe_file['event_types']
    return list_subgroups(ev_group)

@h5_decorator
def count_events(kwe_file, ev_type, ev_name, rec=None):
    return get_events_one_type(kwe_file, ev_type, ev_name, rec=rec).size

@h5_decorator
def get_messages(kwe_file, rec_id=None, node=None):
    rec_table = kwe_file['event_types']['Messages']['events']['recording'][:]
    t_table = kwe_file['event_types']['Messages']['events']['time_samples'][:]
    text_table = kwe_file['event_types']['Messages']['events']['user_data']['Text'][:]
    node_table = kwe_file['event_types']['Messages']['events']['user_data']['nodeID'][:]

    decoder = np.vectorize(lambda x: x.decode('UTF-8'))

    return pd.DataFrame.from_items([('t', t_table),
                                    ('text', decoder(text_table)),
                                    ('rec', rec_table),
                                    ('node', node_table)])
