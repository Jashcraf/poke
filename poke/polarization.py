# dependencies
from poke.poke_math import np,mat_inv_3x3,vector_norm,vector_angle,rotation_3d,vectorAngle,rotation3D
import poke.thinfilms as tf
import poke.poke_math as math
import matplotlib.pyplot as plt

def plot3x3(raybundle,op=np.abs):
    """plots a 3x3 matrix"""

    x = raybundle.xData[0,0]
    y = raybundle.yData[0,0]

    fig,ax = plt.subplots(nrows=3,ncols=3)
    for row in range(3):
        for column in range(3):

            ax[row,column].scatter(x,y,c=op(raybundle.P_total[0][...,row,column]))
    plt.show()


## POLARIZATION RAY TRACING MATH

# Step 1) Compute Fresnel Coefficients
def FresnelCoefficients(aoi,n1,n2,mode='reflect'):

    """Computes Fresnel Coefficients for a single surface interaction

    Parameters
    ----------
    aoi : float or array of floats
        angle of incidence in radians on the interface

    n1 : float 
        complex refractive index of the incident media

    n2 : float
        complex refractive index of the exitant media

    Returns
    -------
    fs, fp: complex floats
        the Fresnel s- and p- coefficients of the surface interaction
    """

    if (mode != 'reflect') and (mode != 'transmit'):
        print('not a valid mode, please use reflect, transmit, or both. Defaulting to reflect')
        mode = 'reflect'

    # ratio of refractive indices
    n = n2/n1

    if mode == 'reflect':

        fs = (np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        fp = (n**2 * np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

    elif mode == 'transmit':

        fs = (2*np.cos(aoi))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        fp = (2*n*np.cos(aoi))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

    return fs,fp

# Step 2) Construct Orthogonal Transfer Matrices
def ConstructOrthogonalTransferMatrices(kin,kout,normal,check_orthogonal=False):
    """Construct the Orthogonal transformations to rotate from global to local coordinates and back again

    Parameters
    ----------
    kin : ndarray
        incident direction cosine vector
    kout : ndarray
        exiting direction cosine vector
    normal : ndarray
        direction cosine vector of the surface normal
    check_orthogonal : bool, optional
        prints the difference of the inverse(O) and transpose(O), should be apprx 0. by default False

    Returns
    -------
    Oinv,Oout : ndarrays
        orthogonal transformation matrices to rotate into the surface local coords (Oinv) and back into global coords (Oout)
    """
    # PL&OS Page 326 Eq 9.5 - 9.7
    # Construct Oin-1 with incident ray, say vectors are row vectors
    kin /= np.linalg.norm(kin) # these were not in chippman and lam - added 03/30/2022
    kout /= np.linalg.norm(kout)

    sin = np.cross(kin,normal)
    sin /= np.linalg.norm(sin) # normalize the s-vector
    pin = np.cross(kin,sin)
    pin /= np.linalg.norm(pin)
    Oinv = np.array([sin,pin,kin])

    sout = sin #np.cross(kout,normal)
    pout = np.cross(kout,sout)
    pout /= np.linalg.norm(pout)
    Oout = np.transpose(np.array([sout,pout,kout]))

    if check_orthogonal == True:
        print('Oinv orthogonality : ',Oinv.transpose() == np.linalg.inv(Oinv))
        print('Oout orthogonality : ',Oout.transpose() == np.linalg.inv(Oout))

    return Oinv,Oout


# Step 3) Create Polarization Ray Trace matrix
def ConstructPRTMatrix(kin,kout,normal,aoi,surfdict,wavelength,ambient_index):

    """Assembles the PRT matrix, relies on the previous two functions

    Parameters
    ----------
    kin : ndarray
        incident direction cosine vector
    kout : ndarray
        exiting direction cosine vector
    normal : ndarray
        direction cosine vector of the surface normal
    aoi : float or array of floats
        angle of incidence in radians on the interface
    surfdict : dict
        dictionary that describe surfaces. Including surface number in raytrace,
        interaction mode, coating, etc.
    wavelength : float
        wavelength of light in meters
    ambient_index : float
        index of the medium that the optical system exists in

    Returns
    -------
    Pmat,J : ndarrays
        Pmat is the polarization ray tracing matrix, J is the same matrix without the orthogonal transformations
    """

    # negate to get to chipman sign convention from zemax sign convention
    normal = -normal

    # Compute the Fresnel coefficients for either transmission OR reflection
    if type(surfdict['coating']) == list:

        # prysm likes films in degress, wavelength in microns, thickness in microns
        rs,ts,rp,tp = tf.ComputeThinFilmCoeffsCLY(surfdict['coating'][:-1],aoi,wavelength,substrate_index=surfdict['coating'][-1])
        
        if surfdict['mode'] == 'reflect':
            fs = rs
            fp = rp * np.exp(-1j*np.pi)  # The Thin Film Correction
        if surfdict['mode'] == 'transmit':
            fs = ts
            fp = tp

    # Single surface coefficients
    else:

        fs,fp = FresnelCoefficients(aoi,ambient_index,surfdict['coating'],mode=surfdict['mode'])
       

    # Compute the orthogonal transfer matrices
    Oinv,Oout = ConstructOrthogonalTransferMatrices(kin,kout,normal)

    # Compute the Jones matrix
    J = np.array([[fs,0,0],[0,fp,0],[0,0,1]])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])

    # Compute the Polarization Ray Tracing Matrix
    Pmat = Oout @ J @ Oinv
    Omat = Oout @ B @ Oinv # The parallel transport matrix, return when ready to implement. This will matter for berry phase

    # This returns the polarization ray tracing matrix but I'm not 100% sure its in the coordinate system of the Jones Pupil
    return Pmat,J#,Omat

def GlobalToLocalCoordinates(Pmat,kin,k,a,exit_x,check_orthogonal=False):

    """Use the double pole basis to compute the local coordinate system of the Jones pupil
    Chipman, Lam, Young, from Ch 11 : The Jones Pupil

    Parameters
    ----------
    Pmat : ndarray
        Pmat is the polarization ray tracing matrix
    kin : ndarray
        incident direction cosine vector at the entrance pupil
    kout : ndarray
        exiting direction cosine vector at the exit pupil
    a : ndarray
        vector in global coordinates describing the antipole direction
    exit_x : ndarray
        vector in global coordinates describing the direction that should be the 
        "local x" direction
    check_orthogonal : bool, optional
        prints the difference of the inverse(O) and transpose(O), should be apprx 0. by default False

    Returns
    -------
    J : ndarray
        shape 3 x 3 ndarray containing the Jones pupil of the optical system. The elements
        Jtot[0,2], Jtot[1,2], Jtot[2,0], Jtot[2,1] should be zero.
        Jtot[-1,-1] should be 1
    """

    
    # Double Pole Coordinate System, requires a rotation about an axis
    # Wikipedia article seems to disagree with CLY Example 11.4
    # Default entrance pupil in Zemax. Note that this assumes the stop is at the first surface
    xin = np.array([1.,0.,0.])
    xin /= np.linalg.norm(xin)
    yin = np.cross(kin,xin)
    yin /= np.linalg.norm(yin)
    O_e = np.array([[xin[0],yin[0],kin[0]],
                    [xin[1],yin[1],kin[1]],
                    [xin[2],yin[2],kin[2]]])

    # Compute Exit Pupil Basis Vectors
    # For arbitrary k each ray will have it's own pair of basis vectors
    r = np.cross(k,a)
    r /= np.linalg.norm(r)
    th = -math.vectorAngle(k,a)
    R = math.rotation3D(th,r)

    # Local basis vectors
    xout = exit_x
    yout = np.cross(a,xout)
    yout /= np.linalg.norm(yout)
    x = R @ xout
    x /= np.linalg.norm(x)
    y = R @ yout
    y /= np.linalg.norm(y)

    O_x = np.array([[x[0],y[0],k[0]],
                    [x[1],y[1],k[1]],
                    [x[2],y[2],k[2]]])

    # Check orthogonality
    if check_orthogonal == True:
        print('O_x difference = ')
        print(O_x.transpose() - np.linalg.inv(O_x))
        print('O_e difference = ')
        print(O_e.transpose() - np.linalg.inv(O_e))

    J = np.linalg.inv(O_x) @ Pmat @ O_e

    return J

# broadcastable functions
def orthogonal_transofrmation_matrices(kin,kout,normal):

    # ensure wave vectors are normalized
    kin = kin / vector_norm(kin)[...,np.newaxis]
    kout = kout / vector_norm(kout)[...,np.newaxis]

    # get s-basis vector
    sin = np.cross(kin,normal)
    sin = sin / vector_norm(sin)[...,np.newaxis]

    # get p-basis vector
    pin = np.cross(kin,sin)
    pin = pin / vector_norm(pin)[...,np.newaxis]

    # Assemble Oinv
    Oinv = np.array([sin,pin,kin])
    Oinv = np.moveaxis(Oinv,-1,0)
    if Oinv.ndim >2:
        for i in range(Oinv.ndim - 2):
            Oinv = np.moveaxis(Oinv,-1,0)
    Oinv = np.swapaxes(Oinv,-1,-2) # take the transpose/inverse

    # outgoing basis vectors
    sout = sin
    pout = np.cross(kout,sout)
    pout = pout / vector_norm(pout)[...,np.newaxis]
    Oout = np.array([sout,pout,kout])
    Oout = np.moveaxis(Oout,-1,0)
    if Oout.ndim >2:
        for i in range(Oout.ndim - 2):
            Oout = np.moveaxis(Oout,-1,0)
    # Oout = np.moveaxis(Oout,0,-1)

    return Oinv,Oout

def prt_matrix(kin,kout,normal,aoi,surfdict,wavelength,ambient_index):
    """prt matrix for a single surface

    Parameters
    ----------
    kin : _type_
        _description_
    kout : _type_
        _description_
    normal : _type_
        _description_
    aoi : _type_
        _description_
    surfdict : _type_
        _description_
    wavelength : _type_
        _description_
    ambient_index : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    normal = -normal

    if type(surfdict['coating']) == list:

        # prysm likes films in degress, wavelength in microns, thickness in microns
        rs,ts,rp,tp = tf.ComputeThinFilmCoeffsCLY(surfdict['coating'][:-1],aoi,wavelength,substrate_index=surfdict['coating'][-1])
        
        if surfdict['mode'] == 'reflect':
            fs = rs
            fp = rp * np.exp(-1j*np.pi)  # The Thin Film Correction

        if surfdict['mode'] == 'transmit':
            fs = ts
            fp = tp
    elif type(surfdict['coating']) == np.ndarray: # assumes the film is defined with first index as fs,fp
        
        fs = surfdict['coating'][0]
        fp = surfdict['coating'][1]

    elif callable(surfdict['coating']): # check if a function
        fs,fp = surfdict['coating'](aoi)

    else:

        fs,fp = FresnelCoefficients(aoi,ambient_index,surfdict['coating'],mode=surfdict['mode'])

    Oinv,Oout = orthogonal_transofrmation_matrices(kin,kout,normal)

    # Compute the Jones matrix and parallel transport matrix
    zeros = np.zeros(fs.shape)
    ones = np.ones(fs.shape)
    J = np.asarray([[fs,zeros,zeros],
                    [zeros,fp,zeros],
                    [zeros,zeros,ones]])
    B = np.asarray([[1,0,0],[0,1,0],[0,0,1]])

    # dimensions need to be appropriate
    if J.ndim > 2:
        for _ in range(J.ndim-2):
            J = np.moveaxis(J,-1,0)

    # compute PRT matrix and orthogonal transformation
    Pmat = Oout @ J @ Oinv
    Qmat = Oout @ B @ Oinv # test if this broadcasts

    return Pmat,J,Qmat

def system_prt_matrices(aoi,kin,kout,norm,surfaces,wavelength,ambient_index):

    P = []
    J = []
    Q = []

    
    for i,surfdict in enumerate(surfaces):

        kisurf = np.moveaxis(kin[i],-1,0)
        kosurf = np.moveaxis(kout[i],-1,0)
        normsurf = np.moveaxis(norm[i],-1,0)
        aoisurf = np.moveaxis(aoi[i],-1,0)
        
        Pmat,Jmat,Qmat = prt_matrix(kisurf,kosurf,normsurf,aoisurf,surfdict,wavelength,ambient_index)
        P.append(Pmat)
        J.append(Jmat)
        Q.append(Qmat)

    return P,J,Q

def total_prt_matrix(P,Q):

    for i,(p,q) in enumerate(zip(P,Q)):

        if i == 0:
            Ptot = p
            Qtot = q
        
        else:
            Ptot = p @ Ptot
            Qtot = q @ Qtot
    
    return Ptot,Qtot

def global_to_local_coordinates(P,kin,k,a,exit_x,Q=None):
    """Use the double pole basis to compute the local coordinate system of the Jones pupil.
    Vectorized to perform on arrays of arbitrary shape, assuming the PRT matrix is in the last
    two dimensions.
    Chipman, Lam, Young, from Ch 11 : The Jones Pupil

    Parameters
    ----------
    Pmat : ndarray
        Pmat is the polarization ray tracing matrix
    kin : ndarray
        incident direction cosine vector at the entrance pupil. Generally an ND array where
        the vector is in the last dimension.
    kout : ndarray
        exiting direction cosine vector at the exit pupil. Generally an ND array where
        the vector is in the last dimension.
    a : ndarray
        vector in global coordinates describing the antipole direction
    exit_x : ndarray
        vector in global coordinates describing the direction that should be the 
        "local x" direction
    Q : Parallel Transport matrix
        the non-polarizing PRT matrix, used to account for geometric transformations

    Returns
    -------
    J : ndarray
        shape 3 x 3 ndarray containing the Jones pupil of the optical system. The elements
        Jtot[0,2], Jtot[1,2], Jtot[2,0], Jtot[2,1] should be zero.
        Jtot[-1,-1] should be 1
    """

    # Double Pole Coordinate System, requires a rotation about an axis
    # Wikipedia article seems to disagree with CLY Example 11.4
    # Default entrance pupil in Zemax. Note that this assumes the stop is at the first surface
    kin = np.moveaxis(kin,-1,0)
    k = np.moveaxis(k,-1,0)
    xin = np.array([1.,0.,0.])
    xin = xin / vector_norm(xin)[...,np.newaxis]
    xin = np.broadcast_to(xin,kin.shape)
    yin = np.cross(kin,xin)
    yin = yin / vector_norm(yin)[...,np.newaxis]
    yin = np.broadcast_to(yin,kin.shape)
    O_e = np.array([[xin[...,0],yin[...,0],kin[...,0]],
                    [xin[...,1],yin[...,1],kin[...,1]],
                    [xin[...,2],yin[...,2],kin[...,2]]])
    O_e = np.moveaxis(O_e,-1,0)

    # Compute Exit Pupil Basis Vectors
    # For arbitrary k each ray will have it's own pair of basis vectors
    r = np.cross(k,a)
    r = r / vector_norm(r)[...,np.newaxis] # match shapes
    th = -vector_angle(k,a)
    R = rotation_3d(th,r)

    # Local basis vectors
    xout = exit_x
    yout = np.cross(a,xout)
    yout /= vector_norm(yout)

    # add axes to match shapes
    xout = xout
    yout = yout
    x = R @ xout
    y = R @ yout

    O_x = np.array([[x[...,0],y[...,0],k[...,0]],
                    [x[...,1],y[...,1],k[...,1]],
                    [x[...,2],y[...,2],k[...,2]]])
    
    O_x = np.moveaxis(O_x,-1,0)

    # apply proper retardance correction
    if type(Q) == np.ndarray:
        P =  mat_inv_3x3(Q) @ P

    J = mat_inv_3x3(O_x) @ P @ O_e

    return J

def JonesToMueller(Jones):

    """Converts a Jones matrix to a Mueller matrix

    Parameters
    ----------
    Jones : 2x2 ndarray
        Jones matrix to convert to a mueller matrix

    Returns
    -------
    M
        Mueller matrix from Jones matrix
    """

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,1j,-1j,0]])

    U *= np.sqrt(1/2)

    M = np.real(U @ (np.kron(np.conj(Jones),Jones)) @ np.linalg.inv(U))

    return M

def MuellerToJones(M):

    """Converts Mueller matrix to a relative Jones matrix. Phase aberration is relative to the Pxx component.

    Returns
    -------
    J : 2x2 ndarray
        Jones matrix from Mueller matrix calculation
    """

    "CLY Eq. 6.112"
    "Untested"

    print('warning : This operation looses global phase')

    pxx = np.sqrt((M[0,0] + M[0,1] + M[1,0] + M[1,1])/2)
    pxy = np.sqrt((M[0,0] - M[0,1] + M[1,0] - M[1,1])/2)
    pyx = np.sqrt((M[0,0] + M[0,1] - M[1,0] - M[1,1])/2)
    pyy = np.sqrt((M[0,0] - M[0,1] - M[1,0] + M[1,1])/2)

    txx = 0 # This phase is not determined
    txy = -np.arctan((M[0,3]+M[1,3])/(M[0,2]+M[1,2]))
    tyx = np.arctan((M[3,0]+M[3,1])/(M[2,0]+M[2,1]))
    tyy = np.arctan((M[3,2]-M[2,3])/(M[2,2]+M[3,3]))

    J = np.array([[pxx*np.exp(-1j*txx),pxy*np.exp(-1j*txy)],
                  [pyx*np.exp(-1j*tyx),pyy*np.exp(-1j*tyy)]])

    return J
    
# def ComputeDRFromAOI(aoi,n1,n2,mode='reflection'):

#     """Computes diattenuation and retardance from angle of incidence

#     Parameters
#     ----------
#     aoi : float or array of floats
#         angle of incidence in radians on the interface

#     n1 : float 
#         complex refractive index of the incident media

#     n2 : float
#         complex refractive index of the exitant media

#     mode : str, reflection or transmission
#         path to trace 

#     Returns
#     -------
#     diattenuation, retardance : floats
#         real valued diattenuation and retardance
#     """

#     fs,fp = FresnelCoefficients(aoi,n1,n2,mode=mode)
    
#     diattenuation = (np.abs(fs)**2 - np.abs(fp)**2)/(np.abs(fs)**2 + np.abs(fp)**2)
#     retardance = np.angle(fs) - np.angle(fp)
    
#     return diattenuation,retardance

# def ComputePauliCoefficients(J):
#     """Computes the pauli coefficients of J

#     Parameters
#     ----------
#     J : ndarray
#         Jones matrix

#     Returns
#     -------
#     c0,c1,c2,c3 : floats
#         Pauli spin matrix coefficients
    
#     """

#     # Isotropic Plate
#     c0 = np.trace(J @ math.PauliSpinMatrix(0))
#     c1 = np.trace(J @ math.PauliSpinMatrix(1))
#     c2 = np.trace(J @ math.PauliSpinMatrix(2))
#     c3 = np.trace(J @ math.PauliSpinMatrix(3))
    
#     return c0,c1,c2,c3

""" Functions to add later """

# def ComputeDRFromJones(J):

#     from scipy.linalg import polar
#     from numpy.linalg import eig,svd

#     evals,evecs = eig(J) # will give us the rotations to quiver
#     W,D,Vh = svd(J) # gives the diattenuation, retardance
#     diattenuation = (np.max(D)**2 - np.min(D)**2)/(np.max(D)**2 + np.min(D)**2) # CLY 5.102
#     U,P = polar(J)
#     # U = W @ np.linalg.inv(Vh)
#     uval,uvec = eig(U)
#     retardance = np.abs(np.angle(uval[0])-np.angle(uval[1])) # CLY 5.81

#     return evecs,diattenuation,retardance

# def ComputeDRFromPRT(P):
    
#     "Yun et al"
    
#     from scipy.linalg import svd,eig
#     #print(P)
#     W,D,Vh = svd(P)
#     eigvals,eigvecs = eig(P)
    
#     print(np.abs(eigvals))
    
    
#     #print(W)
#     #print(D)
#     #print(Vh)
    
#     # singular values given in descending order
#     L1 = D[1]
#     L2 = D[2]
#     #print(L1)
#     #print(L2)
#     diattenuation = (np.abs(L1)**2 - np.abs(L2)**2)/(np.abs(L1)**2 + np.abs(L2)**2)
#     retardance = np.angle(eigvals[2]) - np.angle(eigvals[1])
    
#     return diattenuation,retardance#,Vh[1,:],Vh[2,:]

# def DiattenuationAndRetardancdFromPauli(J):
    
#     c0,c1,c2,c3 = ComputePauliCoefficients(J)
#     c1 /= c0
#     c2 /= c0
#     c3 /= c0
    
#     amp = np.abs(c0)
#     phase = np.angle(c0)
    
#     linear_diattenuation_hv = np.real(c1)
#     linear_retardance_hv = np.imag(c1)
    
#     linear_diattenuation_45 = np.real(c2)
#     linear_retardance_45 = np.imag(c2)
    
#     circular_diattenuation = np.real(c3)
#     circular_retardance = np.imag(c3)
    
#     diattenuation = [linear_diattenuation_hv,linear_diattenuation_45,circular_diattenuation]
#     retardance = [linear_retardance_hv,linear_retardance_45,circular_retardance]
    
#     return amp,phase,diattenuation,retardance
