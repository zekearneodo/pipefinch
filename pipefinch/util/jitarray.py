from numba import jit
import numpy as np


@jit(nopython=True, parallel=True)
def substract_mean_col(x: np.array) -> np.ndarray:
    """ remove offset of a 2d array along axis 1
    Arguments:
        x {np.array} -- [n_i, n_j] array
    
    Returns:
        np.ndarray -- [n_i, n_j] array, with the mean of each column substracted along it
    """
    # remove offset
    n_chan, n_sample = x.shape
    y = np.empty(x.shape)
    for i in np.arange(n_chan):
        y[i, :] = x[i, :] - np.mean(x[i, :])
    return y

@jit(nopython=True, parallel=True)
def collect_frames(x: np.array, row_starts: np.array, row_span: np.int, cols:np.array,
                   dtype=np.int16) -> np.ndarray:
    n_col = cols.size
    n_row = row_starts.size
    
    frames_arr = np.empty((n_row, row_span, n_col), dtype)
    #print('input {}'.format(x.shape))
    #print(frames_arr.shape)
    for i_frame in np.arange(n_row):
        row_start = row_starts[i_frame]
        row_end = int(row_start + row_span)
        for i_col in np.arange(n_col):
            col = cols[i_col]
            #print(i_col)
            #print('{}-{}'.format(row_start, row_end))
            frames_arr[i_frame, :, i_col] = x[row_start: row_end, col]
    return frames_arr
    
def collect_frames_fast(kwd_file, recs_list, starts, span, chan_list):
    recs = np.unique(recs_list)
    all_frames_list = []
    for i_rec, rec in tqdm(enumerate(recs), total=recs.size):
        starts_from_rec = starts[recs_list == rec]
        dset = get_data_set(kwd_file, rec)
        n_samples = dset.shape[0]
        #logger.info('starts_from_rec {}'.format(starts_from_rec.dtype))
        starts_from_rec = int_convert(starts_from_rec, limit=int64_limit - span)
        valid_starts = starts_from_rec[(starts_from_rec > 0)
                                       & (starts_from_rec + span < n_samples)]
        logger.info('valid starts {}'.format(valid_starts))
        if valid_starts.size < starts_from_rec.size:
            logger.warn('Some frames were out of bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(
                valid_starts.size))

        # get the dataset slices for only the channel list
        this_rec_frames = get_slice_array(dset, valid_starts, span, chan_list)
        all_frames_list.append(this_rec_frames)

    try:
        all_frames_array = np.concatenate(all_frames_list, axis=0)
    except ValueError:
        raise
        # logger.warn('Failed to collect stream frames, return is nan array')
        # zero_dset_shape = get_data_set(kwd_file, rec).shape
        # all_frames_array = np.empty([1, *zero_dset_shape])
        # all_frames_array[:] = np.nan
    return all_frames_array