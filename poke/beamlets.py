import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import time
from poke.poke_math import mat_inv_2x2,eigenvalues_2x2,det_2x2
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

    n = k_ray[0]
    RHS = n @ r_pixel
    RHS = np.broadcast_to(RHS,(r_ray.shape[0],RHS.shape[0],RHS.shape[1]))

    LHS = np.sum(ne.evaluate('n*r_ray'),axis=-1)
    LHS = np.broadcast_to(LHS,(RHS.shape[-1],LHS.shape[0],LHS.shape[1]))
    LHS = np.moveaxis(LHS,0,-1)

    DEN = np.sum(ne.evaluate('n*k_ray'),axis=-1)
    DEN = np.broadcast_to(DEN,(LHS.shape[-1],DEN.shape[0],DEN.shape[1]))
    DEN = np.moveaxis(DEN,0,-1)

    Delta = ne.evaluate('(RHS-LHS)/DEN')
    Delta = Delta[...,np.newaxis]

    return Delta

def propagate_rays_and_transform(r_ray,k_ray,Delta,O):
    """propagate rays in free space 

    Parameters
    ----------
    r_rays : ndarray
        position vectors
    k_rays : ndarray
        direction cosine vectors
    Delta : _type_
        distances to propagate along k_rays

    Returns
    -------
    r_rays,k_rays
        broadcasted r and k rays after propagating
    """

    # swap Delta to match rays
    Delta = np.moveaxis(Delta,-2,0)

    # Now has a new dim 1
    r_ray = ne.evaluate('r_ray + k_ray*Delta')

    r_ray = r_ray[...,np.newaxis]
    k_ray = k_ray[...,np.newaxis]

    # O = np.broadcast_to(O,(r_ray.shape[0],r_ray.shape[2],r_ray.shape[1],3,3))
    # O = np.moveaxis(O,1,2)

    r_ray = O @ r_ray
    k_ray = O @ k_ray

    # consider swapping axes back
    # r_ray =  np.swapaxes(r_ray,0,1)

    # broadcast k_ray
    k_ray = np.broadcast_to(k_ray,r_ray.shape)
    
    return r_ray,k_ray

def differential_matrix_calculation(central_u,central_v,diff_uu,diff_uv,diff_vu,diff_vv,du,dv):
    """computes a sub-matrix of the ray transfer tensor

    diff_ij means a differential ray with initial differential in dimension i, evaluated in j
    diff_xy means the differential ray with an initial dX in the x dimension on the source plane,
    and these are the coordinates of that ray in the y-axis on the detector plane

    Parameters
    ----------
    central_u : numpy.ndarray
        array describing the central rays position or angle in x or l
    central_v : numpy.ndarray
        array describing the central rays position or angle in y or m
    diff_uu : numpy.ndarray
        array describing the differential rays position or angle in x or l
    diff_uv : numpy.ndarray
        array describing the differential rays position or angle in y or m
    diff_vu : numpy.ndarray
        array describing the differential rays position or angle in y or m
    diff_vv : numpy.ndarray
        array describing the differential rays position or angle in y or m
    du : float
        differential on sourc plane in position or angle in x or l
    dv : float
        differential on sourc plane in position or angle in y or m

    Returns
    -------
    numpy.ndarray
        sub-matrix of the ray transfer tensor
    """

    A = ne.evaluate('(diff_uu - central_u)/du') # Axx
    C = ne.evaluate('(diff_uv - central_v)/du') # Axy
    B = ne.evaluate('(diff_vu - central_u)/dv') # Ayx
    D = ne.evaluate('(diff_vv - central_v)/dv') # Ayy
    diffmat = np.moveaxis(np.array([[A,B],[C,D]]),-1,0)
    diffmat = np.moveaxis(diffmat,-1,0)

    return diffmat

def center_transversal_plane(r_pixels,r_ray,O):

    """Centers the coordinate system on the transversal plane

    Returns
    -------
    r
        coordinates of distances from the center of the transversal plane to the pixels
    """
    r_ray = np.moveaxis(r_ray,1,0)
    
    # pre-treat r pixel
    r_pixels = np.broadcast_to(r_pixels,(O.shape[0],*r_pixels.shape))
    r_pixels = np.moveaxis(r_pixels,-1,0)
    r_pixels = r_pixels[...,np.newaxis]

    r_pixels = O @ r_pixels
    r_origin = r_ray[:,0] # skip over pixel dimension to grab the central ray
    r = ne.evaluate('r_pixels-r_origin')
    r = r[...,0] # drop the newaxis used for matmul

    return r

def prop_complex_curvature(Qinv,A,B,C,D):

    NUM = (C + D @ Qinv)
    DEN = mat_inv_2x2(A + B @ Qinv)

    return NUM @ DEN

def transversal_phase(Qpinv,r):

    transversal = (r[...,0]*Qpinv[...,0,0] + r[...,1]*Qpinv[...,1,0])*r[...,0]
    transversal = (transversal + (r[...,0]*Qpinv[...,0,1] + r[...,1]*Qpinv[...,1,1])*r[...,1])/2

    return transversal

def optical_path_and_delta(OPD,Delta):
    OPD = OPD[0] # central ray
    Delta = np.moveaxis(Delta[0,...,0],-1,0) # central ray
    opticalpath = OPD + Delta # grab central ray of OPD

    return opticalpath

def guoy_phase(Qpinv):

    e1,e2 = eigenvalues_2x2(Qpinv)
    guoy = np.arctan((np.real(e1)/np.imag(e1)) + (np.real(e2)/np.imag(e2)))/2

    return guoy

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
    print('computeunit = ',computeunit)
    print(dcoords.shape)
    field = np.zeros([dcoords.shape[1]],dtype=np.complex128)

    # offset detector coordinates by ray centroid
    if use_centroid:
        mean_base = np.mean(np.asarray([xData,yData,zData])[:,0,-1],axis=-1)
        print(mean_base.shape)
        print('centroid at = ',mean_base)
        if np.__name__ == 'jax.numpy': # accomodate for jax quirk
            dcoords.at([...,0]).set(dcoords[...,0] + mean_base[0])
            dcoords.at([...,1]).set(dcoords[...,1] + mean_base[1])
        else:
            dcoords[...,0] = dcoords[...,0] + mean_base[0]
            dcoords[...,1] = dcoords[...,1] + mean_base[1]
    
    t1 = time.perf_counter()
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

        # construct orthogonal transformation
        O = orthogonal_transformation_matrix(k_ray[0],dnorm)

        # get ray positions
        Delta = distance_to_transversal(dcoords,r_ray,k_ray)

        # propagate rays to transversal plane and orthogonal transform
        r_ray,k_ray = propagate_rays_and_transform(r_ray,k_ray,Delta,O)

        # get ray back with the rayset index in front
        r_ray = np.moveaxis(r_ray,1,0)
        k_ray = np.moveaxis(k_ray,1,0)

        # waist rays
        A = differential_matrix_calculation(r_ray[0,...,0,0],r_ray[0,...,1,0], # central ray central_u,v
                                            r_ray[1,...,0,0],r_ray[1,...,1,0], # waist_x diff_uu,uv
                                            r_ray[2,...,0,0],r_ray[2,...,1,0], # waist_y diff_vu,vv
                                            dPx,dPy)
        
        C = differential_matrix_calculation(k_ray[0,...,0,0],k_ray[0,...,1,0], # central ray central_u,v
                                            k_ray[1,...,0,0],k_ray[1,...,1,0], # waist_x diff_uu,uv
                                            k_ray[2,...,0,0],k_ray[2,...,1,0], # waist_y diff_vu,vv
                                            dPx,dPy)
        
        B = differential_matrix_calculation(r_ray[0,...,0,0],r_ray[0,...,1,0], # central ray central_u,v
                                            r_ray[3,...,0,0],r_ray[3,...,1,0], # diverge_x diff_uu,uv
                                            r_ray[4,...,0,0],r_ray[4,...,1,0], # diverge_y diff_vu,vv
                                            dHx,dHy)
        
        D = differential_matrix_calculation(k_ray[0,...,0,0],k_ray[0,...,1,0], # central ray central_u,v
                                            k_ray[3,...,0,0],k_ray[3,...,1,0], # diverge_x diff_uu,uv
                                            k_ray[4,...,0,0],k_ray[4,...,1,0], # diverge_y diff_vu,vv
                                            dHx,dHy)
        del k_ray

        # center pixels on transversal plane
        r = center_transversal_plane(dcoords,r_ray,O)
        del r_ray,O

        # propagate gaussian field
        Qpinv = prop_complex_curvature(Qinv,A,B,C,D)
        del C,D

        # compute amplitude
        Amplitude = 1/(np.sqrt(det_2x2(A + B @ Qpinv)))
        del A,B

        # phase terms
        transversal = -1j*k*transversal_phase(Qpinv,r)
        guoy = 1j*guoy_phase(Qpinv)
        del Qpinv


        opticalpath = -1j*k*optical_path_and_delta(OPD,Delta)
        print('optical path shape = ',opticalpath.shape)
        del OPD,Delta

        # total phasor
        field += np.sum(Amplitude*ne.evaluate('exp(transversal+opticalpath+guoy)'),axis=1)
        del Amplitude,transversal,opticalpath,guoy

        print(f'Finished loop {loop}, took {time.perf_counter()-t1}s, estimated completion in {(time.perf_counter()-t1/(loop+1))*nloops}s')

    return field












        
