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

def plot_state(state, fc=(0.5,0,0.2,0.5)):
    """ Plots a state specified as a 3D boolean array """
    
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    
    ax.voxels(state, facecolors=fc, edgecolors='k')
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']);
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)


def plot_coords(coords):
    """ Plot cubes at the coordinates given by coords (N,3). """
    ranges = np.ptp(coords, 0).astype(int) + 1
    state = np.zeros(ranges, bool)
    coords = (coords - coords.min(0)).astype(int)
    state[coords[:,0], coords[:,1], coords[:,2]] = True
    plot_state(state)
    
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
