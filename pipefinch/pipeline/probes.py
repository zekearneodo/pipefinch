import numpy as np
from typing import Union
# probe_site maps
# each probe is a dict {site: coord}, where site is an int, coord is a 2/3 tuple (x, y, z)
# as in the probe_maping
probe = {}
probe['a1x32-edge-5mm-20-177_H32'] = {'group_0': {i +
    1: (0, i*20) for i in range(32)}}
probe['a1x32-poly3-5mm-25s-177_H32'] = {'group_0': {17: (18, 275),  # column 1
                                        16: (18, 250),
                                        18: (18, 225),
                                        15: (18, 200),
                                        19: (18, 175),
                                        14: (18, 150),
                                        20: (18, 125),
                                        13: (18, 100),
                                        21: (18, 75),
                                        12: (18, 50),
                                        22: (18, 25),
                                        11: (18, 0),
                                        10: (0, 237),  # column 0
                                        9: (0, 212),
                                        8: (0, 187),
                                        7: (0, 162),
                                        6: (0, 137),
                                        5: (0, 112),
                                        4: (0, 87),
                                        3: (0, 62),
                                        2: (0, 37),
                                        1: (0, 12),
                                        23: (36, 237),  # column 3
                                        24: (36, 212),
                                        25: (36, 187),
                                        26: (36, 162),
                                        27: (36, 137),
                                        28: (36, 112),
                                        29: (36, 87),
                                        30: (36, 62),
                                        31: (36, 37),
                                        32: (36, 12)
                                        }
                                        }

# port_site maps
# each port is a dict {port: site}, where port is an int, site is an int
# port is

headstage = {}
# intan headstage with neuronexus h32 connector (omnetics), AND intan software
headstage['intan32-h32'] = {23: 15, 22: 6, 21: 5, 20: 4, 19: 16, 18: 3, 17: 2, 16: 1,
                            15: 32, 14: 31, 13: 30, 12: 17, 11: 29, 10: 28, 9: 27, 8: 18,
                            24: 13, 25: 12, 26: 11, 27: 10, 28: 14, 29: 9, 30: 8, 31: 7,
                            0: 26, 1: 25, 2: 24, 3: 19, 4: 23, 5: 22, 6: 21, 7: 20}

def flatten_probe(prb_dict: dict) -> dict:
    # flatten a probe (merge all chan groups into a single dict of site: coords)
    flat_probe = {k: v for group in prb_dict.values() for k, v in group.items()}
    return flat_probe

def make_map(probe_name: str, chan_list, headstage_name: str='intan32-h32', 
    return_dict: bool=False) -> Union[dict, np.array]:
    """[summary]
    Make a chan: (x, y) coordinate map, taking into account the probe, the headstage, and the
    list of channels used.
    Arguments:
        probe_name {str} -- name of the probe (to lookup the probe from the probe definitions dictionary in probes.py)
        chan_list {[type]} -- list of channels of the probe present in the recordings
    Keyword Arguments:
        headstage_name {str} -- name of the headstage (to lookup the headstage from the hs definitions dictionary in probes.py)
     (default: {'intan32-h32'})
        return_dict {bool} -- whether to return the whole dictionary or just the ordered coordintates (mountainsort style) (default: {False})
    
    Returns:
        Union[dict, np.array] -- either a dictionary (return_dict is True) or a np.array.
        if return_dict is True:
            port_coords_dict = dict{'group_0': {chan_0: (x_0, y_0),
                                                chan_1: (x_1, y_1),
                                                ...
                                                }
                                    'group_1': {...}
                                    }
                             where each group is a shank, for instance.
        else:
            port_coords_array = [[x_0, y_0], [x_1, y_1], ..., [x_n, y_n]] where n are all the channels listed 
            in chan_list.
    """
    # want to retrieve what coordinates are in each port of the headstage
    # headstage is {port: site}, probe is {site: coords}
    a_headstage = headstage[headstage_name]
    a_probe = probe[probe_name]

    port_coords_dict = {}
    for grp_name, grp_dict in a_probe.items():
        port_coords_dict[grp_name] = {port: np.array(grp_dict[site]) for port, site in sorted(
            a_headstage.items()) if port in chan_list}
    if return_dict:
        return port_coords_dict
    else:
        return np.vstack(list(flatten_probe(port_coords_dict).values()))

def site_port_map(probe_name: str, headstage_name: str, only_group=None):
    if only_group is None:
        a_probe = flatten_probe(probe[probe_name])
    else:
        a_probe = probe[probe_name[only_group]]
    a_probe = flatten_probe(probe[probe_name])
    a_headstage = headstage[headstage_name]
    return [h_port for site in a_probe.keys() for h_port, h_site in a_headstage.items() if h_site == site]
