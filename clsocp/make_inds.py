import numpy as np

def make_inds(kvec):
    curr = 0
    block = 0
    inds = []
    while(block < len(kvec)):
        inds.append(list(range(curr, int((curr + kvec[block])))))
        curr = int(curr + kvec[block])
        block += 1
    return inds
