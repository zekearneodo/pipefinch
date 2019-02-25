import numpy as np
# probe_site maps
# each probe is a dict {site: coord}, where site is an int, coord is a 2/3 tuple (x, y, z)
# as in the probe_maping
probe = {}
probe['a1x32-edge-5mm-20-177_H32'] = {i+1: (0, i*20) for i in range(32)}
probe['a1x32-poly3-5mm-25s-177_H32'] = {17: (18, 275),  # column 1
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

# port_site maps
# each port is a dict {port: site}, where port is an int, site is an int
# port is

headstage = {}
# intan headstage with neuronexus h32 connector (omnetics), AND intan software
headstage['intan32-h32'] = {23: 15, 22: 6, 21: 5, 20: 4, 19: 16, 18: 3, 17: 2, 16: 1,
                            15: 32, 14: 31, 13: 30, 12: 17, 11: 29, 10: 28, 9: 27, 8: 18,
                            24: 13, 25: 12, 26: 11, 27: 10, 28: 14, 29: 9, 30: 8, 31: 7,
                            0: 26, 1: 25, 2: 24, 3: 19, 4: 23, 5: 22, 6: 21, 7: 20}

def make_map(probe_name: str, chan_list, headstage_name: str = 'intan32-h32') -> np.array:
    # want to retrieve what coordinates are in each port of the headstage
    # headstage is {port: site}, probe is {site: coords}
    a_probe = probe[probe_name]
    a_headstage = headstage[headstage_name]
    port_coords = {port: np.array(a_probe[site]) for port, site in sorted(
        a_headstage.items()) if port in chan_list}
    return np.vstack(list(port_coords.values()))

def site_port_map(probe_name: str, headstage_name: str):
    a_probe = probe[probe_name]
    a_headstage = headstage[headstage_name]
    return [h_port for site in a_probe.keys() for h_port, h_site in a_headstage.items() if h_site==site]