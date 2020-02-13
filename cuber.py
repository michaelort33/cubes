# -*- coding: utf-8 -*-
import numpy as np
from methods import gen_neighbors, bool_to_ind, get_mapping, convert_coords, plot_board, plot_piece, remove_piece, recolor_piece
from scipy.ndimage.filters import uniform_filter

#%% This function will be called recursively
def make_move(S, plot=False):
    """ Tries to make a valid move on the board in state S (10,10,10). If no valid move is available
    returns False. After moving, calls recursively, if a history is returned, returns the move history
    with the current move appended. If plot is True, will also plot the pieces as it goes. """
    
    #Choose the most constrained cell to target, can also be replaced with a faster heuristic
    density = uniform_filter(S.astype(float))*~S
    target = np.unravel_index(density.argmax(),density.shape)

    #Find the neighborhood occupancy and lookup the valid moves
    x = S[convert_coords(N, target)]
    moves = M[np.where(Z[bool_to_ind(x)])]

    for move in moves:
        location = convert_coords(move,target)
        S[location] = True
        piece = plot_piece(location) #Turn off for speed        
        
        if S[board].all(): # If we've just filled the useable space, we won!
            print("We won!")
            return [location]
        else:
            print(S[board].sum())
        recolor_piece(piece) # Turn off for speed
        result = make_move(S)
        if result: # If a history is returned this has proved successful
            print("Passing the history up the stack...")
            return result.append(location)
        
        # If we made it to here, we're still playing and need to try a better move
        print("Taking back a move to try another")
        S[location] = False
        remove_piece(piece)
    
    # If we made it here, we're out of moves and have to go take something back
    print("I'm on a dead branch, passing the info upstream")
    return False

#%% Setup
N = gen_neighbors()
M = get_mapping()
size = 4
S = np.ones([size+4]*3, bool)
board = np.s_[2:-2,2:-2,2:-2]
S[board] = False
plot_board(size)

#%% Load the lookup table !REPLACE WITH THE ACTUAL LOAD COMMAND WHEN IT'S DONE
Z = np.zeros((256**3, 48))

#%% Play!
result = make_move(S)