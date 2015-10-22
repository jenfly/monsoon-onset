import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
def slice_premidpost(imid, n):
    """Return slices to index for pre, mid, and post composites.

    Parameters
    ----------
    imid : int or list of ints
        Mid-point index of range to define composites.
    n : int
        Size of range to define composites.

    Returns
    -------
    inds : dict of slices (or lists of slices)
        For each i in imid, the slices are:
          'pre' : slice(i - n1 - n, i - n1)
          'mid' : slice(i - n1, i + n2)
          'post' : slice(i + n2, i + n2 + n)
        where n1 = n // 2 and n2 = n-n1
    """

    n1 = n // 2
    n2 = n - n1
    inds = {'pre' : [], 'mid' : [], 'post' : []}
    for i in atm.makelist(imid):
        i0 = int(i - n1 - n)
        i1 = int(i - n1)
        i2 = int(i + n2)
        i3 = int(i + n2 + n)
        inds['pre'].append(slice(i0, i1))
        inds['mid'].append(slice(i1, i2))
        inds['post'].append(slice(i2, i3))

    # Collapse lists if input imid was a single value
    if len(atm.makelist(imid)) == 1:
        for key in inds:
            inds[key] = inds[key][0]

    return inds


# ----------------------------------------------------------------------
def composite_premidpost(data, onset, n):
    """Return composite data fields relative to onset index.

    """

    # Get metadata from xray.DataArray

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
        comp[key] = comp[key].mean(axis=1)

    # Pack metadata

    return comp
