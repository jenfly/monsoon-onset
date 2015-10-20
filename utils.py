import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import matplotlib.pyplot as plt

import atmos as atm
import merra

def slice_premidpost(imid, n):
    n1 = n // 2
    n2 = n - n1
    inds = {}
    inds['pre'] = slice(imid - n1 - n, imid - n1)
    inds['mid'] = slice(imid - n1, imid + n2)
    inds['post'] = slice(imid + n2, imid + n2 + n)
    return inds


def composite_premidpost(data, onset, n):
    dims = list(data.shape)
    dims[1] = n
    comp = {}
    comp['pre'] = np.zeros(dims, dtype=float)
    comp['mid'] = np.zeros(dims, dtype=float)
    comp['post'] = np.zeros(dims, dtype=float)

    for y, d in enumerate(onset):
        inds = slice_premidpost(d - 1, n)
        for key in comp:
            comp[key][y] = data[y, inds[key]]
    for key in comp:
        comp[key][y] = comp[key][y].mean(axis=1)
    return comp