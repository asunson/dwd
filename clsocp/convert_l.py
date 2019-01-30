import numpy as np

def convert_l(kvec, t):
    newkvec = []
    newtype = []
    for i in range(len(t)):
        if t[i] == 'l':
            newkvec = newkvec + [1 for j in range(int(kvec[i]))]
            newtype = newtype + ['q' for j in range(int(kvec[i]))]
        elif t[i] == 'q':
            newkvec.append(int(kvec[i]))
            newtype.append(t[i])
    return newkvec, newtype

    
    