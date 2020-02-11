# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import product, permutations, chain


# get the array of 48 possible orientations for a block

def get_mapping():
    """ Generates a table of the 48 unique placements of a block on a cell"""
    coords = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    coords = [coords - i for i in coords]

    x = [[0, 90, 180, 270]] * 3

    all_coords = []

    for i in product(*x):
        r = R.from_euler('xyz', i, degrees=True)
        for one_coord in coords:
            all_coords.append(np.sort(r.apply(one_coord).round(), axis=0))

    unique_coordinates = np.unique(np.array(all_coords), axis=0)

    return unique_coordinates


def gen_neighbors():
    """ Generate a table of the 24 reachable neighbors of a cell """
    r = [-1,0,1]
    return list(filter(lambda x: 0<np.abs(x).sum()<3, chain(product(r,r,r),permutations([0,0,2]))))


def plot_state(state, fc=(0.5,0,0.2,0.5)):
    """ Plots a state specified as a 3D boolean array """
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    ax.voxels(state, facecolors=fc, edgecolors='k')