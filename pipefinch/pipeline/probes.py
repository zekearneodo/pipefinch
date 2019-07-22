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

# generated with make_cambridge_prb notebook
probe['assy-236-h5'] = {'group_0': {1: (20, 130),
                                    2: (20, 10),
                                    3: (0, 40),
                                    4: (20, 110),
                                    5: (0, 80),
                                    6: (20, 170),
                                    7: (20, 270),
                                    8: (0, 220),
                                    9: (0, 380),
                                    10: (20, 430),
                                    11: (20, 330),
                                    12: (0, 480),
                                    13: (0, 540),
                                    14: (20, 610),
                                    15: (20, 550),
                                    16: (0, 580),
                                    17: (0, 60),
                                    18: (0, 300),
                                    19: (0, 140),
                                    20: (20, 190),
                                    21: (20, 90),
                                    22: (20, 250),
                                    23: (0, 280),
                                    24: (0, 400),
                                    25: (20, 390),
                                    26: (20, 410),
                                    27: (20, 450),
                                    28: (20, 350),
                                    29: (0, 500),
                                    30: (20, 490),
                                    31: (20, 590),
                                    32: (0, 440),
                                    33: (20, 50),
                                    34: (20, 310),
                                    35: (0, 120),
                                    36: (0, 180),
                                    37: (20, 30),
                                    38: (0, 260),
                                    39: (0, 160),
                                    40: (0, 240),
                                    41: (20, 210),
                                    42: (0, 420),
                                    43: (20, 370),
                                    44: (0, 340),
                                    45: (0, 320),
                                    46: (20, 510),
                                    47: (20, 630),
                                    48: (0, 600),
                                    49: (20, 150),
                                    50: (0, 0),
                                    51: (0, 20),
                                    52: (0, 100),
                                    53: (20, 70),
                                    54: (20, 290),
                                    55: (0, 200),
                                    56: (20, 230),
                                    57: (0, 360),
                                    58: (20, 470),
                                    59: (0, 460),
                                    60: (0, 520),
                                    61: (0, 560),
                                    62: (20, 570),
                                    63: (20, 530),
                                    64: (0, 620)
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

headstage['cambridge-64'] = {0: 2,
 1: 3,
 2: 5,
 3: 6,
 4: 7,
 5: 8,
 6: 9,
 7: 26,
 8: 25,
 9: 18,
 10: 19,
 11: 20,
 12: 21,
 13: 22,
 14: 23,
 15: 33,
 16: 24,
 17: 34,
 18: 50,
 19: 35,
 20: 36,
 21: 51,
 22: 53,
 23: 37,
 24: 54,
 25: 55,
 26: 38,
 27: 56,
 28: 39,
 29: 40,
 30: 57,
 31: 58,
 32: 59,
 33: 41,
 34: 60,
 35: 42,
 36: 61,
 37: 43,
 38: 62,
 39: 44,
 40: 63,
 41: 27,
 42: 64,
 43: 28,
 44: 46,
 45: 29,
 46: 47,
 47: 31,
 48: 48,
 49: 30,
 50: 45,
 51: 32,
 52: 10,
 53: 11,
 54: 12,
 55: 13,
 56: 15,
 57: 14,
 58: 16,
 59: 17,
 60: 49,
 61: 1,
 62: 52,
 63: 4}

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
