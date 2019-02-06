# some objects to do quick stuff on events
import numpy as np
import h5py
import logging
import pandas as pd

from pipefinch.h5tools.kwik import kwdfunctions as kwdf
logger = logging.getLogger('pipefinch.h5tools.kwik.event')


class Event:
    name = None
    start = None
    rec = None
    end = None
    meta = None
    sampling_rate = None
    where_event = None
    has_event = None
    get_table_function = None

    data = None
    datagroup = None
    rec_datasets = dict()
    event_ch_idx = None
    kwik_meta_pd = None
    ch_names = None

    def __init__(self, name, h5_file=None):
        self.name = name
        self.h5_file = h5_file


class TTL(Event):
    def __init__(self, name: str, ch_name: str, h5_file: h5py.File):
        Event.__init__(self, name, h5_file=h5_file)

        # the names of the datasets whithin each rec where the relevant data is

        # get the file metadata
        kwik_meta_pd = kwdf.get_all_rec_meta(h5_file)

        # lookup the digital channel within the table of dig channels
        all_dig_ch = kwdf.get_all_chan_names(kwik_meta_pd,
                                             block='digital',
                                             chan_filt=np.array(['DIN']))
        # event_ch_idx = kwdf.find_chan_names_idx(all_dig_ch,
        #                                         np.array([ch_name]))
        # assert event_ch_idx.size == 1, 'Should have found 1 occurrence of ch {},but found {}'.format(
        #     ch_name, event_ch_idx.size)
        #self.event_ch_idx = event_ch_idx

        self.kwik_meta_pd = kwik_meta_pd
        self.rec_datasets = {'edge': 'dig_edge',
                             'ch_idx': 'dig_edge_ch',
                             't': 'dig_edge_t'}
        self.ch_names = all_dig_ch
        self.event_pd = pd.DataFrame(np.empty([0, len(self.rec_datasets.keys()) + 1]),
                                     columns=( ['rec'] + list(self.rec_datasets.keys()) ),
                                     dtype=np.int)
        self.fill_event_pd()

    # fill a pandas dataframe with all the events
    def fill_event_pd(self):
        all_rec = self.kwik_meta_pd['name'].values
        for rec in all_rec:
            r_group = self.h5_file['/recordings/{}'.format(rec)]
            # the datasets
            ev_arr = np.hstack([r_group['{}'.format(x)]
                                for x in self.rec_datasets.values()])
            rec_pd = pd.DataFrame(ev_arr, columns=self.rec_datasets.keys(),
            dtype=np.int)
            rec_pd['rec'] = rec
            self.event_pd = self.event_pd.append(rec_pd, sort=False)
        self.event_pd['ch'] = self.event_pd['ch_idx'].apply(lambda x: self.ch_names[x])
        return self.event_pd
        
