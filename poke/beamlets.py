import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

def orthogonal_transformation_matrix(n,normal):
    """generates the orthogonal transformation to the transversal plane

    Parameters
    ----------
    n : N x 3 vector, typically k_ray[0]
        vector normal to the transversal plane, typically the central ray of a beamlet
    normal : N x 3 vector, typically (0,0,1)
        local surface normal of the detector plane

    Returns
    -------
    O : Nx3x3 ndarray
        orthogonal transformation matrix
    """
    l = np.cross(n,-normal)
    m = np.cross(n,l)
    
    O = np.asarray([[l[...,0],l[...,1],l[...,2]],
                    [m[...,0],m[...,1],m[...,2]],
                    [n[...,0],n[...,1],n[...,2]]])
    
    O = np.moveaxis(O,-1,0)

    return O

def distance_to_transversal(r_pixel,r_ray,k_ray):

    # n = k_ray[0]

    return Delta

def beamlet_decomposition_field(xData,yData,zData,mData,lData,nData,opd,dPx,dPy,dHx,dHy,dcoords,dnorm,
                                wavelength=1.65e-6,nloops=32,use_centroid=True):
    """computes the coherent beamlet decomposition field from ray data

    Parameters
    ----------
    xData : numpy.ndarray
        the x position coordinates of the rays at the final field of evaluation
    yData : numpy.ndarray
        the y position coordinates of the rays at the final field of evaluation
    zData : numpy.ndarray
        the z position coordinates of the rays at the final field of evaluation
    mData : numpy.ndarray
        the l position coordinates of the rays at the final field of evaluation
    lData : numpy.ndarray
        the m position coordinates of the rays at the final field of evaluation
    nData : numpy.ndarray
        the n position coordinates of the rays at the final field of evaluation
    opd : numpy.ndarray
        the OPD of the rays at the final field of evaluation
    dPx : float
        The ray differential in position used to compute the ABCD matrix
    dPy : float
        The ray differential in position used to compute the ABCD matrix
    dHx : float
        The ray differential in direction cosine used to compute the ABCD matrix
    dHy : float
        The ray differential in direction cosine used to compute the ABCD matrix
    dcoords : N x 3 numpy.ndarray
        coordinates of detector on final field of evaluation
    dnorm : N x 3 numpy.ndarray or, a single vector
        surface normal of detector on final field of evaluation
    """

    # Set up complex curvature
    wo = dPx
    zr = (np.pi * wo * wo)/wavelength
    qinv = 1/(1j*zr)
    Qinv = np.asarray([[qinv,0],[0,qinv]])
    k = 2*np.pi/wavelength

    # Break up the problem
    nbeams = nData[:,-1].shape[1]
    computeunit = int(nbeams/nloops)
    field = np.zeros([dcoords.shape[0]],dtype=np.complex128).ravel()

    # offset detector coordinates by ray centroid
    if use_centroid:
        mean_base = np.mean(np.asarray([xData,yData,zData])[:,0,-1],axis=-1)
        if np.__name__ == 'jax.numpy': # accomodate for jax quirk
            dcoords.at([...,0]).set(dcoords[...,0] + mean_base[0])
            dcoords.at([...,1]).set(dcoords[...,1] + mean_base[1])
        else:
            dcoords[...,0] = dcoords[...,0] + mean_base[0]
            dcoords[...,1] = dcoords[...,1] + mean_base[1]
    
    for loop in range(nloops):

        if loop < nloops-1:

            xEnd = xData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]
            yEnd = yData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]
            zEnd = zData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]
            lEnd = lData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]
            mEnd = mData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]
            nEnd = nData[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]

            OPD = opd[:,-1,int(computeunit*loop):int(computeunit*(loop+1))]

        elif loop == nloops-1:

            xEnd = xData[:,-1,int(computeunit*loop):]
            yEnd = yData[:,-1,int(computeunit*loop):]
            zEnd = zData[:,-1,int(computeunit*loop):]
            lEnd = lData[:,-1,int(computeunit*loop):]
            mEnd = mData[:,-1,int(computeunit*loop):]
            nEnd = nData[:,-1,int(computeunit*loop):]
            OPD = opd[:,-1,int(computeunit*loop):]

        # construct ray postions and directions
        r_ray = np.moveaxis(np.asarray([xEnd,yEnd,zEnd]),0,-1)
        del xEnd,yEnd,zEnd
        k_ray = np.moveaxis(np.asarray([lEnd,mEnd,nEnd]),0,-1)
        del lEnd,mEnd,nEnd

        O = orthogonal_transformation_matrix(k_ray[0],dnorm)


        
