# imports
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools as iter


# get the array of 48 possible orientations for a block

def get_mapping():
    coords = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    coords = [coords - i for i in coords]

    x = [[0, 90, 180, 270]] * 3

    all_coords = []

    for i in iter.product(*x):
        r = R.from_euler('xyz', i, degrees=True)
        for one_coord in coords:
            all_coords.append(np.sort(r.apply(one_coord).round(), axis=0))

    unique_coordinates = np.unique(np.array(all_coords), axis=0)

    return unique_coordinates
