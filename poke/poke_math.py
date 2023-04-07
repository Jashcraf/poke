import numpy as np

def det_2x2(array):
    a = array[...,0,0]
    b = array[...,0,1]
    c = array[...,1,0]
    d = array[...,1,1]

    det = (a*d - b*c)

    return det

def mat_inv_2x2(array):

    a = array[...,0,0]
    b = array[...,0,1]
    c = array[...,1,0]
    d = array[...,1,1]

    det = (a*d - b*c)

    matinv = np.array([[d,-b],[-c,a]]) / det
    if matinv.ndim > 2:
        for i in range(matinv.ndim-2):
            matinv = np.moveaxis(matinv,-1,0)
        
    return matinv

def mat_inv_3x3(array):

    a = array[...,0,0] # row 1
    b = array[...,0,1]
    c = array[...,0,2]
    
    d = array[...,1,0] # row 2
    e = array[...,1,1]
    f = array[...,1,2]
    
    g = array[...,2,0] # row 3
    h = array[...,2,1]
    i = array[...,2,2]

    # determine cofactor elements
    ac = e*i - f*h
    bc = -(d*i - f*g)
    cc = d*h - e*g
    dc = -(b*i - c*h)
    ec = a*i - c*g
    fc = -(a*h - b*g)
    gc = b*f - c*e
    hc = -(a*f - c*d)
    ic = a*e - b*d

    # get determinant
    det = a*ac + b*bc + c*cc # second term's negative is included in cofactor term
    det = det[...,np.newaxis,np.newaxis]

    # Assemble adjucate matrix (transpose of cofactor)
    arrayinv = np.asarray([[ac,bc,cc],
                           [dc,ec,fc],
                           [gc,hc,ic]]).T / det
    
    return arrayinv

def eigenvalues_2x2(array):
    """ Computes the eigenvalues of a 2x2 matrix using a trick
    Parameters
    ----------
    array : numpy.ndarray
        a N x 2 x 2 array that we are computing the eigenvalues of
    Returns
    -------
    e1, e2 : floats of shape N
        The eigenvalues of the array
    """

    a = array[...,0,0]
    b = array[...,0,1]
    c = array[...,1,0]
    d = array[...,1,1]

    determinant = (a*d - b*c)
    mean_ondiag = (a+d)/2
    e1 = mean_ondiag + np.sqrt(mean_ondiag**2 - determinant)
    e2 = mean_ondiag - np.sqrt(mean_ondiag**2 - determinant)

    return e1,e2 

def vector_norm(vector):
    vx = vector[...,0] * vector[...,0]
    vy = vector[...,1] * vector[...,1]
    vz = vector[...,2] * vector[...,2]

    return np.sqrt(vx + vy + vz)

def vector_angle(u,v):
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
    u = u/(vector_norm(u)[...,np.newaxis])
    v = v/(vector_norm(v)[...,np.newaxis])

    dot = np.sum(u*v,axis=-1)
    angles = np.zeros_like(dot)

    # Make exceptions for angles turning around
    if dot.any() < 0:
        angles[dot < 0] = np.pi - 2*np.arcsin(vector_norm(-v-u)/2)
    if dot.any() >= 0:
        angles[dot >= 0] = 2*np.arcsin(vector_norm(v-u)/2)
    
    return angles

def rotation_3d(angle,axis):
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
    mat = np.array([[(1-c)*axis[...,0]**2 + c, (1-c)*axis[...,0]*axis[...,1] - s*axis[...,2], (1-c)*axis[...,0]*axis[...,2] + s*axis[...,1]],
                    [(1-c)*axis[...,1]*axis[...,0] + s*axis[...,2], (1-c)*axis[...,1]**2 + c, (1-c)*axis[...,1]*axis[...,2] - s*axis[...,0]],
                    [(1-c)*axis[...,2]*axis[...,0] - s*axis[...,1], (1-c)*axis[...,1]*axis[...,2] + s*axis[...,0], (1-c)*axis[...,2]**2 + c]])
    if mat.ndim > 2:
        for i in range(mat.ndim-2):
            mat = np.moveaxis(mat,-1,0)
    return mat


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
    u = u/vector_norm(u)
    v = v/vector_norm(v)

    if u@v<0: # dot product
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
    