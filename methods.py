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
            unsorted = r.apply(one_coord).round().astype(int)
            all_coords.append(unsorted[np.lexsort(unsorted.T)])

    unique_coordinates = np.unique(np.array(all_coords), axis=0)

    return unique_coordinates


def gen_neighbors():
    """ Generate a table of the 24 reachable neighbors of a cell """
    r = [-2,-1,0,1,2]
    return np.array(list(filter(lambda x: 0<np.abs(x).sum()<3, product(r,r,r))))

def plot_board(size=6, fc=(1,1,0.8,1), ec=(0.2,0.2,0.2,1)):
    """ Plots the initial board given an empty state array """
    box = np.ones([size+2]*3, bool)
    box[2:,2:,2:] = False
    fig = plt.figure()
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
    plt.pause(0.1)
    return vox
    
def remove_piece(handles):
    """ Remove a piece by deleteting all voxels in the handle dictionary returned by plot_piece"""
    [x.remove() for x in handles.values()]
    
def recolor_piece(handles, color=(0.8,0.8,0.8,0.7)):
    """ Recolors a piece given by the handle dictionary returned by plot_piece"""
    [x.set_facecolor(color) for x in handles.values()]
    
def get_valid_moves(M,N,x):
    """ Get all allowable moves in M (48,4,3), that don't collide with the 
    occupancy coordinates in N (24,3), given the occupancy boolean x (24,)"""
    return np.array([not(m[:, None] == N[x]).all(-1).any() for m in M])

def gen_mapmap(M,N):
    """ Get a Moves Neighbors map in a single boolean matrix 48 moves (rows) by 24 neighbors (cols).
    Each cell is true if the move in its row uses the neighbor in its column."""
    return (M[:,:,None]==N).all(-1).any(1)

def get_valid_moves2(X,x):
    """ Get all allowable moves in X (48,24), given the occupancy boolean x (24,)"""
    return ~X[:,x].any(-1)

def bool_to_ind(x):
    """ Convert a booleann vector x into it's equivalent index in base 10 """
    return int(''.join([str(int(b)) for b in x]),2)

def convert_coords(coords, target):
    """ Converts coordinates (N,3) given with respect to a local target (3,) cell into global 
    coordinates suitable for indexing into the full state."""
    return tuple((coords + np.asarray(target)).T)

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

    lookup_table = []
    for one_state in tqdm(possible_states[:1000]):
        blocked_points = neighbors[[i==1 for i in one_state]]

        moves = []
        for i in legal_moves:
            legality_boolean = []
            for k in blocked_points:
                if(check_block_for_point(i,k)):
                    legality_boolean.append(1)
                else:
                    legality_boolean.append(0)
            if sum(legality_boolean)==0:
                moves.append(1)
            else:
                moves.append(0)
        lookup_table.append([moves,one_state])

    return lookup_table
