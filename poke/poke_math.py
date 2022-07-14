import numpy as np

def MatmulList(array1,array2):

    # only works for square matrices
    out = np.empty(array1.shape,dtype='complex128')

    for i in range(array1.shape[-1]):
        out[:,:,i] = array1[:,:,i] @ array2[:,:,i]

    return out