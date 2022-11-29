import numpy as np

def MatmulList(array1,array2):
    """Multiplies two lists of matrices. This is unnecessary because numpy already broadcasts multiplications
    TODO : remove all dependencies on this function and replace with matmul with appropriate broadcasting dimensions
    """

    # only works for square matrices
    out = np.empty(array1.shape,dtype='complex128')

    for i in range(array1.shape[-1]):
        out[:,:,i] = array1[:,:,i] @ array2[:,:,i]

    return out

"Vector Operations from Quinn Jarecki"
import numpy as np
import cmath as cm

def rotation3D(angle,axis):
    """Rotation matrix about an axis by an angle

    Parameters
    ----------
    angle : float
        rotation angle in radians
    axis : ndarray
        shape 3 vector in cartesian coordinates to rotate about

    Returns
    -------
    mat : ndarray
        rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[(1-c)*axis[0]**2 + c, (1-c)*axis[0]*axis[1] - s*axis[2], (1-c)*axis[0]*axis[2] + s*axis[1]],
                    [(1-c)*axis[1]*axis[0] + s*axis[2], (1-c)*axis[1]**2 + c, (1-c)*axis[1]*axis[2] - s*axis[0]],
                    [(1-c)*axis[2]*axis[0] - s*axis[1], (1-c)*axis[1]*axis[2] + s*axis[0], (1-c)*axis[2]**2 + c]])
    return mat

def vectorAngle(u,v):
    """computes the vector angle between two vectors

    Parameters
    ----------
    u : ndarray
        shape 3 vector
    v : ndarray
        shape 3 vector

    Returns
    -------
    ndarray
        vector of angle between u and v in x, y, z in radians
    """
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    if u@v<0:
        return np.pi - 2*np.arcsin(np.linalg.norm(-v-u)/2)
    else:
        return 2*np.arcsin(np.linalg.norm(v-u)/2)

def PauliSpinMatrix(i):

    """Returns the pauli spin matrix of index i

    Parameters
    ----------
    i : int
        pauli spin matrix index. Can be 0, 1, 2, or 3

    Returns
    -------
    ndarray
        Pauli spin matrix of the corresponding index
    """

    if i == 0:
    
        return np.array([[1,0],[0,1]])
        
    if i == 1:
    
        return np.array([[1,0],[0,-1]])
        
    if i == 2:
        return np.array([[0,1],[1,0]])
        
    if i == 3:
    
        return np.array([[0,-1j],[1j,0]])
    