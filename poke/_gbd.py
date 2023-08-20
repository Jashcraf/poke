import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne

import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

def disp_array(array,cmap='viridis'):
    plt.figure()
    plt.imshow(array,cmap=cmap)
    plt.colorbar()
    plt.show()

def Matmulvec(x2,y2,M,x1,y1):
    """Multiplies vectors r2 = [x2,y2], r1 = [x1,y1], with matrix M in
    return r2^T @ M r1

    Parameters
    ----------
    x2 : _type_
        _description_
    y2 : _type_
        _description_
    M : _type_
        _description_
    x1 : _type_
        _description_
    y1 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    return (x2*M[0,0] + y2*M[1,0])*x1 + (x2*M[0,1] + y2*M[1,1])*y1

def ComputeGouyPhase(Q):

    """Computes the Guoy phase of the complex curvature matrix Q

    Parameters
    ----------
    Q : numpy.ndarray
        2x2 complex curvature matrix. Follows numpy's matrix broadcasting rules.
    
    """
    eigvals = np.linalg.eigvals(Q)
    q1,q2 = eigvals[0],eigvals[1]

    gouy = .5*(np.arctan(np.real(q1)/np.imag(q1)) + np.arctan(np.real(q2)/np.imag(q2)))

    return gouy


def EvalField(xData,yData,zData,lData,mData,nData,opd,dPx,dPy,dHx,dHy,detsize,npix,normal=np.array([0.,0.,1.]),wavelength=1.65e-6):

    """

    Paramters
    ---------

    x/y/zData : numpy.ndarray
        the position coordinates of the rays at the final field of evaluation

    l/m/nData : numpy.ndarray
        the icrection cosine coordinates of the rays at the final field of evaluation

    opd : numpy.ndarray
        Optical path difference for each ray
    
    dPx/y : float
        The ray differential in position used to compute the ABCD matrix

    dHx/y : float
        The ray differential in direction cosine used to compute the ABCD matrix

    detsize : float
        The side length of the square detector to evaluate the field on

    npix : int
        The number of pixels across detsize

    normal : numpy.ndarray
        Direction cosine vector describing the normal of the detector (assuming a planar detector). Defaults to [0,0,1]
    
    wavelength : float
        Distance in meters corresponding to the optical wavelength of interest

    Returns
    -------
    Field : complex128 numpy.ndarray
        The simulated scalar field at the detector using GBD

    
    This will evaluate the field as gaussian beamlets at the last surface the rays are traced to

    RECALL: The ray data shape is [len(raysets),len(surflist),maxrays] from TraceThroughZOS

    1) read ray data
    2) compute transversal plane basis vectors
    3) find plane normal to z and including detector coordinate
    4) Find where ray intersects plane
    5) Propagate to the plane
    6) Compute the ABCD matrix on the plane
    7) Compute field
    
    TODO: Make more flexible, this assumes that all beamlets are fundamental mode gaussians
    """

    """Set up complex curvature
    """
    wo = dPx
    zr = (np.pi*wo**2)/wavelength
    qinv = 1/(1j*zr)

    Qinv = np.array([[qinv,0],
                     [0,qinv]])

    k = 2*np.pi/wavelength

    """Set up Detector
    """

    # Set up detector vector
    x = np.linspace(-detsize/2,detsize/2,npix)
    x,y = np.meshgrid(x,x)
    r0 = np.array([x.ravel(),y.ravel(),0*x.ravel()]) # detector plane is the z = 0 plane
    print('r0 shape = ',r0.shape)

    """
    Read Ray Data
    """

    ## 1) Read Ray Data, now shape [len(raysets),maxrays]

    xEnd = xData[:,-1] # Positions
    yEnd = yData[:,-1]
    zEnd = zData[:,-1]
    lEnd = lData[:,-1] # Direction Cosines
    mEnd = mData[:,-1]
    nEnd = nData[:,-1]


    # Try to keep variables in the parlance of Ashcraft et al 2022
    # These are now shape 3 x len(raysets) x maxrays, so we move the spatial index to the last axis
    # These are the central rays
    rdet = np.moveaxis(np.array([xEnd,yEnd,zEnd]),0,-1)
    mean_base = np.mean(rdet[0],axis=0)
    rdet -= mean_base
    print('centroid at ',mean_base)
    kdet = np.moveaxis(np.array([lEnd,mEnd,nEnd]),0,-1)

    ## 2) Compute Transversal Plane Basis Vectors, only use central ray
    n = kdet[0]

    l = np.cross(n,-normal) # normal is eta in Ashcraft et al 2022
    lx = l[...,0]
    ly = l[...,1]
    lz = l[...,2]

    lnorm = np.sqrt(lx**2 + ly**2 + lz**2) # np.linalg.norm(l,axis=-1)
    l[:,0] /= lnorm
    l[:,1] /= lnorm
    l[:,2] /= lnorm
    m = np.cross(n,l)

    # print('Determine Norms (should be 1)')
    # print('-------------------------')
    # print(np.linalg.norm(l,axis=-1))
    # print(np.linalg.norm(m,axis=-1))
    # print(np.linalg.norm(n,axis=-1))
    # print('-------------------------')

    # Wrong shape for matmul, gotta moveaxis
    O = np.array([[l[...,0],l[...,1],l[...,2]],
                  [m[...,0],m[...,1],m[...,2]],
                  [n[...,0],n[...,1],n[...,2]]]) 

    # Clear the prior variables
    del l, m

    print('O before the reshape = ',O.shape)
    O = np.moveaxis(O,-1,0)

    ## Compute the Position to Update
    RHS = n @ r0 # n dot r0, broadcast for every pixel and beamlet
    RHS = np.broadcast_to(RHS,(rdet.shape[0],RHS.shape[0],RHS.shape[1]))

    LHS = np.sum(n*rdet,axis=-1) # n dot rdet
    LHS = np.broadcast_to(LHS,(RHS.shape[-1],LHS.shape[0],LHS.shape[1]))
    LHS = np.moveaxis(LHS,0,-1)

    DEN = np.sum(n*kdet,axis=-1) # n dot kdet
    DEN = np.broadcast_to(DEN,(LHS.shape[-1],DEN.shape[0],DEN.shape[1]))
    DEN = np.moveaxis(DEN,0,-1)

    Delta = (RHS-LHS)/DEN
    Delta = Delta[...,np.newaxis]
    del RHS,LHS,DEN,n

    kdet = np.broadcast_to(kdet,(Delta.shape[-2],kdet.shape[0],kdet.shape[1],kdet.shape[2]))
    kdet = np.moveaxis(kdet,0,-2)
    rdet = np.broadcast_to(rdet,(Delta.shape[-2],rdet.shape[0],rdet.shape[1],rdet.shape[2]))
    rdet = np.moveaxis(rdet,0,-2)
    O = np.broadcast_to(O,(rdet.shape[0],rdet.shape[2],rdet.shape[1],3,3))
    O = np.moveaxis(O,1,2)

    # Get a bunch of updated ray positions, remember the broadcasting rules
    rdetprime = rdet + kdet*Delta
    
    rdetprime = rdetprime[...,np.newaxis]
    
    rtransprime = O @ rdetprime

    # rayset, beamlet, pixel, dimension, extra
    kdet = kdet[...,np.newaxis]
    ktrans = O @ kdet
    r0 = np.moveaxis(r0,0,-1)[...,np.newaxis]
    ktrans = np.broadcast_to(ktrans,rtransprime.shape)

    del kdet,rdet,rdetprime


    ## Now compute the ray transfer matrix from the data
    central_r = rtransprime[0]
    central_k = ktrans[0]

    waistx_r = rtransprime[1]
    waistx_k = ktrans[1]

    waisty_r = rtransprime[2]
    waisty_k = ktrans[2]

    divergex_r = rtransprime[3]
    divergex_k = ktrans[3]

    divergey_r = rtransprime[4]
    divergey_k = ktrans[4]


    Axx = (waistx_r[...,0,0] - central_r[...,0,0])/dPx
    Ayx = (waistx_r[...,1,0] - central_r[...,1,0])/dPx
    Cxx = (waistx_k[...,0,0] - central_k[...,0,0])/dPx
    Cyx = (waistx_k[...,1,0] - central_k[...,1,0])/dPx

    Axy = (waisty_r[...,0,0] - central_r[...,0,0])/dPy
    Ayy = (waisty_r[...,1,0] - central_r[...,1,0])/dPy
    Cxy = (waisty_k[...,0,0] - central_k[...,0,0])/dPy
    Cyy = (waisty_k[...,1,0] - central_k[...,1,0])/dPy

    Bxx = (divergex_r[...,0,0] - central_r[...,0,0])/dHx
    Byx = (divergex_r[...,1,0] - central_r[...,1,0])/dHx
    Dxx = (divergex_k[...,0,0] - central_k[...,0,0])/dHx
    Dyx = (divergex_k[...,1,0] - central_k[...,1,0])/dHx

    Bxy = (divergey_r[...,0,0] - central_r[...,0,0])/dHy
    Byy = (divergey_r[...,1,0] - central_r[...,1,0])/dHy
    Dxy = (divergey_k[...,0,0] - central_k[...,0,0])/dHy
    Dyy = (divergey_k[...,1,0] - central_k[...,1,0])/dHy

    del waistx_k,waistx_r,waisty_k,waisty_r,divergex_k,divergex_r,divergey_r,divergey_k

    ABCD = np.array([[Axx,Axy,Bxx,Bxy],
                     [Ayx,Ayy,Byx,Byy],
                     [Cxx,Cxy,Dxx,Dxy],
                     [Cyx,Cyy,Dyx,Dyy]])

    ABCD = np.moveaxis(ABCD,0,-1)
    ABCD = np.moveaxis(ABCD,0,-1)

    r0prime = O @ r0
    r = r0prime - central_r

    del r0,r0prime

    # Grab the central
    r = r[0,...,0]

    """
    # For Displaying the Ray Transfer Matrix
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    beamid = 0
    fig,ax = plt.subplots(nrows=4,ncols=4,figsize=[10,10])
    for i in range(4):
        for j in range(4):

            im = ax[i,j].scatter(r[beamid,:,0],r[beamid,:,1],c=ABCD[beamid,:,i,j])
            div = make_axes_locatable(ax[i,j])
            cax = div.append_axes("right",size='5%',pad="2%")
            cb = fig.colorbar(im,cax=cax)
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
    """


    """
    Propagate the Q Parameter
    """
    A = ABCD[...,0:2,0:2]
    B = ABCD[...,0:2,2:4]
    C = ABCD[...,2:4,0:2]
    D = ABCD[...,2:4,2:4]

    del ABCD, Axx,Axy,Ayx,Ayy,Bxx,Bxy,Byx,Byy,Cxx,Cxy,Cyx,Cyy,Dxx,Dxy,Dyx,Dyy
    Num = (C + np.matmul(D , Qinv))
    Den =  np.linalg.inv(A + np.matmul(B , Qinv))
    Qpinv = np.matmul(Num,Den)
    del Num,Den

    Amplitude = 1/(np.sqrt(np.linalg.det(A + B @ Qpinv)))

    transversal = (-1j*k/2)*((r[...,0]*Qpinv[...,0,0] + r[...,1]*Qpinv[...,1,0])*r[...,0] + (r[...,0]*Qpinv[...,0,1] + r[...,1]*Qpinv[...,1,1])*r[...,1])
    opticalpath = (-1j*k)*(opd[0,-1] + np.moveaxis(Delta[0,...,0],0,-1))
    opticalpath = np.moveaxis(opticalpath,0,-1)

    result,vecs = np.linalg.eig(Qpinv)
    eig1 = result[...,0]
    eig2 = result[...,1]
    guoy = 1j*0.5*(np.arctan(np.real(eig1)/np.imag(eig1)) + np.arctan(np.real(eig2)/np.imag(eig2)))
    del result,vecs,eig1,eig2

    Phase = transversal+opticalpath+guoy
    del transversal,opd,guoy,Delta,Qpinv,r

    """
    Compute the Gaussian Field
    """

    Field = Amplitude*ne.evaluate('exp(Phase)')

    # print('Field Shape = ',Field.shape)
    
    # Coherent Superposition
    Field = np.sum(Field,axis=0)
    # print('Coherent field shape = ',Field.shape)

    return Field