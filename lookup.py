# -*- coding: utf-8 -*-
from methods import gen_neighbors, gen_mapmap, get_mapping, get_valid_moves2
from itertools import product
import numpy as np
from tqdm import tqdm 

N = gen_neighbors()
M = get_mapping()
X=gen_mapmap(M,N)

inds=np.array(list(product(*[[False, True]]*24))) # Every possible neighborhood
Z=np.zeros([len(inds),48],bool) # Preallocate lookup table for speed
for i,x in tqdm(enumerate(inds)):
    Z[i]=get_valid_moves2(X,x)
    
np.save('Z.npy', Z)