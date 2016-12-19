"""
Adapted from the official scipy implementation available at
https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py
"""

import numpy as np

def rankdata(a):
    arr = np.ravel(np.asarray(a))
    algo = 'quicksort'
    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)


def score_function(y_true, y_pred):
    # Rank each vector
    ar = rankdata(y_true)
    br = rankdata(y_pred)
    # Compute the rho value
    rs = np.corrcoef(ar, br)
    # Return the result
    if rs.shape == (2, 2):
        return rs[1, 0]
    else:
        return rs
