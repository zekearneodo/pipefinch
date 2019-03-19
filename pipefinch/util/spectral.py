from scipy import signal
import numpy as np


def ms_spectrogram(x, s_f, n_window=512, step_ms=1):

    # the overlap is the size of the window minus the smples in a msec
    msec_samples = int(s_f * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * s_f

    # Make the spectrogram
    f, t, Sxx = signal.spectrogram(x, s_f,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.gaussian(n_window, sigma),
                                   scaling='spectrum')

    Sxx[[Sxx < np.max((Sxx) * 0.000055)]] = 1

    return f, t, Sxx
