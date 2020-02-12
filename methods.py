# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import product, permutations
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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
            unsorted = r.apply(one_coord).round()
            all_coords.append(unsorted[np.lexsort(unsorted.T)])

    unique_coordinates = np.unique(np.array(all_coords), axis=0)

    return unique_coordinates


def gen_neighbors():
    """ Generate a table of the 24 reachable neighbors of a cell """
    r = [-2,-1,0,1,2]
    return list(filter(lambda x: 0<np.abs(x).sum()<3, product(r,r,r)))

def plot_board(size=6, fc=(1,1,0.8,1), ec=(0.2,0.2,0.2,1)):
    """ Plots the initial board given an empty state array """
    box = np.ones([size+2]*3, bool)
    box[2:,2:,2:] = False
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    ax.view_init(20,15)
    vox = ax.voxels(box, facecolors=fc, edgecolors=ec)
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']);
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)
    ax.autoscale(False)
    return vox

def plot_piece(location, fc=(0.4,0,0.4,1), ec=(0.2,0.2,0.2,1)):
    """ Plots a single piece in the given location (3 tuple of x,y,z coordinates)"""
    piece = np.zeros([max(x) + 1 for x in location], bool)
    piece[location] = True
    ax = plt.gca()
    vox = ax.voxels(piece, facecolors=fc, edgecolors=ec)
    plt.pause(0.01)
    return vox
    
def remove_piece(handles):
    """ Remove a piece by deleteting all voxels in the handle dictionary returned by plot_piece"""
    [x.remove() for x in handles.values()]
    
def recolor_piece(handles, color='0.8'):
    """ Recolors a piece given by the handle dictionary returned by plot_piece"""
    [x.set_facecolor(color) for x in handles.values()]
    
def get_valid_moves(M,N,x):
    """ Get all allowable moves in M (48,4,3), that don't collide with the 
    occupancy coordinates in N (24,3), given the occupancy boolean x (24,)"""
    return np.array([not(m[:, None] == N[x]).all(-1).any() for m in M])

def bool_to_ind(x):
    """ Convert a booleann vector x into it's equivalent index in base 10 """
    return np.r_[np.packbits(x)[::-1],np.uint8(0)].view(np.uint32)

def check_block_for_point(block, point):

    for i in block:
        if all(i == point):
            return True
    return False

def gen_legal_moves():
    legal_moves = get_mapping().astype(int)
    neighbors = np.array(gen_neighbors())
    x = [[1,0]] * 24
    possible_states = list(product(*x))

    moves = []
    for one_state in tqdm(possible_states):
        blocked_points = neighbors[one_state==1]
        legality_boolean = []

        for i in legal_moves:
            for k in blocked_points:
                if(check_block_for_point(i,k)):
                    legality_boolean.append(0)
                else:
                    legality_boolean.append(1)

        moves.append(legality_boolean)

    lookup_table = np.array((possible_states, moves)).T

    return lookup_table
