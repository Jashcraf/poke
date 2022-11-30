import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne

# Config numexpr numthreads
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
from numba import njit


def Matmulvec(x2,y2,M,x1,y1):

    return (x2*M[0,0] + y2*M[1,0])*x1 + (x2*M[0,1] + y2*M[1,1])*y1

def ComputeGouyPhase(Q):


    eigvals = np.linalg.eigvals(Q)
    q1,q2 = eigvals[0],eigvals[1]

    gouy = .5*(np.arctan(np.real(q1)/np.imag(q1)) + np.arctan(np.real(q2)/np.imag(q2)))

    return gouy


"""Let's try out a more efficient beamlet propagation algorithm that's outside of the inner for loop,
and we try to loop over nbeamlets?
"""

def EvalField(xData,yData,zData,lData,mData,nData,opd,dPx,dPy,dHx,dHy,detsize,npix,normal=np.array([0.,0.,1.]),wavelength=1e-6):

    """sudo code
    
    This will evaluate the field as gaussian beamlets at the last surface the rays are traced to

    RECALL: The ray data shape is [len(raysets),len(surflist),maxrays] from TraceThroughZOS

    1) read ray data
    2) compute transversal plane basis vectors
    3) find plane normal to z and including detector coordinate
    4) Find where ray intersects plane
    5) Propagate to the plane
    6) Compute the ABCD matrix on the plane
    7) Compute field
    """

    """
    Set up complex curvature
    TODO: Make more flexible, this assumes that all beamlets are fundamental mode gaussians
    """
    wo = dPx
    qinv = -1j*wavelength/(np.pi*wo)

    Qinv = np.array([[qinv,0],
                     [0,qinv]])

    k = 2*np.pi/wavelength

    """Set up Detector
    """

    # Set up detector vector
    x = np.linspace(-detsize/2,detsize/2,npix)
    x,y = np.meshgrid(x,x)
    r0 = np.array([x.ravel(),y.ravel(),0*x.ravel()]) # detector plane is the z = 0 plane


    """
    Read Ray Data
    """

    ## 1) Read Ray Data, now shape [len(raysets),maxrays]
    xStart = xData[:,0] # Positions
    yStart = yData[:,0]
    zStart = zData[:,0]
    lStart = lData[:,0] # Direction Cosines
    mStart = mData[:,0]
    nStart = nData[:,0]

    xEnd = xData[:,-1] # Positions
    yEnd = yData[:,-1]
    zEnd = zData[:,-1]
    lEnd = lData[:,-1] # Direction Cosines
    mEnd = mData[:,-1]
    nEnd = nData[:,-1]

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=[7,7])
    # # plt.subplot(121)
    # plt.title('Position')
    # plt.scatter(xData[0,-1],yData[0,-1],label='zeroth')
    # # plt.scatter(xData[1,-1],yData[1,-1],label='first')
    # # plt.scatter(xData[2,-1],yData[2,-1],label='second')
    # plt.scatter(xData[3,-1],yData[3,-1],label='third')
    # plt.scatter(xData[4,-1],yData[4,-1],label='fourt')
    # # plt.colorbar()
    # plt.legend()
    # # plt.subplot(122)
    # # plt.title('Angle')
    # # plt.scatter(lEnd,mEnd,c=nEnd)
    # # plt.colorbar()
    # plt.show()

    # Try to keep variables in the parlance of Ashcraft et al 2022
    # These are now shape 3 x len(raysets) x maxrays, so we move the spatial index to the last axis
    # These are the central rays
    rdet = np.moveaxis(np.array([xEnd,yEnd,zEnd]),0,-1)
    kdet = np.moveaxis(np.array([lEnd,mEnd,nEnd]),0,-1)

    # print('kdetector shape = ',np.array([lEnd,mEnd,nEnd]).shape)
    print(kdet[0])

    ## 2) Compute Transversal Plane Basis Vectors, only use central ray
    n = kdet[0]
    # print('n shape = ',n.shape)

    l = np.cross(n,-normal) # normal is eta in Ashcraft et al 2022
    # print('l shape = ',l.shape)
    print(l)
    lx = l[...,0]
    ly = l[...,1]
    lz = l[...,2]

    lnorm = np.sqrt(lx**2 + ly**2 + lz**2) # np.linalg.norm(l,axis=-1)

    # print(lnorm)

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

    # print('O before the reshape = ',O.shape)
    O = np.moveaxis(O,-1,0)
    # print('O after the reshape = ',O.shape)

    # print('transposition ----------')
    print(np.transpose(O[0]))
    # print('inverse ----------------')
    print(np.linalg.inv(O[0]))

    # O = np.linalg.inv(O)

    # Why don't we try do a loopless computation anyway? We are already here

    # print('r0 shape = ',r0.shape)
    # print('n shape = ',n.shape)

    ## Compute the Position to Update
    RHS = n @ r0 # n dot r0, broadcast for every pixel and beamlet
    # print('RHS shape = ',RHS.shape)

    # RHS = np.moveaxis(RHS,-1,0)
    RHS = np.broadcast_to(RHS,(rdet.shape[0],RHS.shape[0],RHS.shape[1]))
    # print('RHS shape = ',RHS.shape)
    # RHS = np.moveaxis(RHS,0,1)
    # print('N @ R0 = ',RHS.shape)

    LHS = np.sum(n*rdet,axis=-1) # n dot rdet
    LHS = np.broadcast_to(LHS,(RHS.shape[-1],LHS.shape[0],LHS.shape[1]))
    LHS = np.moveaxis(LHS,0,-1)
    # print('LHS Shape = ',LHS.shape)

    DEN = np.sum(n*kdet,axis=-1) # n dot kdet
    DEN = np.broadcast_to(DEN,(LHS.shape[-1],DEN.shape[0],DEN.shape[1]))
    DEN = np.moveaxis(DEN,0,-1)
    # print('DEN shape = ',DEN.shape)
    # print(DEN)

    Delta = (RHS-LHS)/DEN
    Delta = Delta[...,np.newaxis]

    # print('Delta shape = ',Delta.shape)

    # Gotta reshape k
    kdet = np.broadcast_to(kdet,(Delta.shape[-2],kdet.shape[0],kdet.shape[1],kdet.shape[2]))
    kdet = np.moveaxis(kdet,0,-2)
    rdet = np.broadcast_to(rdet,(Delta.shape[-2],rdet.shape[0],rdet.shape[1],rdet.shape[2]))
    rdet = np.moveaxis(rdet,0,-2)
    O = np.broadcast_to(O,(rdet.shape[0],rdet.shape[2],rdet.shape[1],3,3))
    O = np.moveaxis(O,1,2)
    # print('kdet shape = ',kdet.shape)
    # print('rdet shape = ',rdet.shape)
    # print('O shape = ',O.shape)

    # Get a bunch of updated ray positions, remember the broadcasting rules
    rdetprime = rdet + kdet*Delta
    # rdetprime = rdetprime[...,np.newaxis] 
    # kdet = kdet[...,np.newaxis]
    # print('updated rdet shape = ',rdetprime.shape)
    # print('updated kdet shape = ',kdet.shape)
    rdetprime = rdetprime[...,np.newaxis]
    rtransprime = O @ rdetprime
    kdet = kdet[...,np.newaxis]
    ktrans = O @ kdet
    r0 = np.moveaxis(r0,0,-1)[...,np.newaxis]
    ktrans = np.broadcast_to(ktrans,rtransprime.shape)
    # print('rtrans shape = ',rtransprime.shape)
    # print('ktrans shape = ',ktrans.shape)

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

    # print('shape central r = ',central_r.shape)
    # print('central k shape = ',central_k.shape)

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

    ABCD = np.array([[Axx,Axy,Bxx,Bxy],
                     [Ayx,Ayy,Byx,Byy],
                     [Cxx,Cxy,Dxx,Dxy],
                     [Cyx,Cyy,Dyx,Dyy]])

    ABCD = np.moveaxis(ABCD,0,-1)
    ABCD = np.moveaxis(ABCD,0,-1)


    # print('ABCD shape = ',ABCD.shape)
    # print('r0 shape = ',r0.shape)
    # print('central r shape = ',central_r.shape)

    r0prime = O @ r0
    r = r0prime - central_r
    # print('radial coordinate shape = ',r.shape)
    # dr = r0-rdetprime
    # print(r[...,2,0])
    # print(r[...,2,0])

    # print('r shape = ',r.shape)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=[10,5])
    # plt.subplot(121)
    # plt.title('Transformed Coordinates')
    # plt.scatter(r[...,0,0],r[...,1,0],c=r[...,2,0])
    # plt.colorbar()
    # plt.subplot(122)
    # plt.title('Untransformed')
    # plt.scatter(dr[...,0,0],dr[...,1,0],c=dr[...,2,0])
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(kdet[0,...,0,0],kdet[0,...,1,0],kdet[0,...,2,0],label='kdet')
    # ax.scatter(ktrans[0,...,0,0],ktrans[0,...,1,0],ktrans[0,...,2,0],label='ktrans')
    # plt.show()

    """
    Propagate the Q Parameter
    """

    A = ABCD[...,0:2,0:2]
    B = ABCD[...,0:2,2:4]
    C = ABCD[...,2:4,0:2]
    D = ABCD[...,2:4,2:4]

    Qpinv = (C + D @ Qinv) @ np.linalg.inv(A + B @ Qinv)
    Amplitude = 1/(np.sqrt(np.linalg.det(A + B @ Qinv)))

    print('Qpinv shape = ',Qpinv.shape)
    print('Amplitude shape = ',Amplitude.shape)

    # the first index of this is shape 5, so I think we just pick the first because they are all the same?
    transversal = (-1j*k/2)*((r[0,...,0,0]*Qpinv[...,0,0] + r[0,...,1,0]*Qpinv[...,1,0])*r[0,...,0,0] + (r[0,...,0,0]*Qpinv[...,0,1] + r[0,...,1,0]*Qpinv[...,1,1])*r[0,...,1,0])
    print('transverse phase shape = ',transversal.shape)

    # The shape of this is weird
    print('OPD shape = ',opd.shape)
    print('Central delta shape = ',Delta[0].shape)
    opticalpath = (-1j*k)*(opd[0,-1] + np.moveaxis(Delta[0,...,0],0,-1))
    opticalpath = np.moveaxis(opticalpath,0,-1)
    print('Optical path shape = ',opticalpath.shape)
    # print(transversal)
    # print(opticalpath)

    Phase = transversal+opticalpath

    print('Phase shape = ',Phase.shape)

    """
    Compute the Gaussian Field
    """

    Field = Amplitude*np.exp(Phase)

    print('Field Shape = ',Field.shape)
    
    # Coherent Superposition
    Field = np.sum(Field,axis=0)
    print('Coherent field shape = ',Field.shape)

    return Field




    


    
    
def MarchRayfront(raybundle,dis,surf=-1):
    
    # Positions
    x = raybundle.xData[surf]
    y = raybundle.yData[surf]
    z = raybundle.zData[surf]
    
    # Angles
    l = raybundle.lData[surf]
    m = raybundle.mData[surf]
    n = raybundle.nData[surf]
    
    # arrange into vectors
    r = np.array([x,y,z])
    k = np.array([l,m,n])
    
    # propagate
    r_prime = r + k*dis
    
    # change the positions
    raybundle.xData[surf] = r_prime[0,:]
    raybundle.yData[surf] = r_prime[1,:]
    raybundle.zData[surf] = r_prime[2,:]

    return raybundle
    
# def ComputeFinitePropagation(raybundle,detsize,dnorm=np.array([0,0,1]),surf=-1):
    
    # # Positions
    # x = raybundle.xData[surf]
    # y = raybundle.yData[surf]
    # z = raybundle.zData[surf]
    # r = np.array([x,y,z])
    
    # # Angles
    # l = raybundle.lData[surf]
    # m = raybundle.mData[surf]
    # n = raybundle.nData[surf]
    # k = np.array([l,m,n])
    
    # dnorm = np.array([np.ones(x.shape)*dnorm[0],
                      # np.ones(x.shape)*dnorm[1],
                      # np.ones(x.shape)*dnorm[2]])
    
    # th = np.arccos(np.sum(k*dnorm,axis=0)) # angles at image plane
    # R = np.sqrt(x**2 + y**2 + z**2) # positions at image plane
    
    # d_plus = (detsize/2 + R)*np.sin(th)
    # d_minus = (detsize/2 - R)*np.sin(th)
    
    # return d_plus,d_minus 
    
def PropagateQparam(Qinv,sys):
    A = sys[0:2,0:2]
    B = sys[0:2,2:4]
    C = sys[2:4,0:2]
    D = sys[2:4,2:4]

    # print(A.shape)
    # print(B.shape)
    # print(C.shape)
    # print(D.shape)
    
    # Step 2 - Propagate Complex Beam Parameter
    Qp_n = (C + D @ Qinv)
    Qp_d = np.linalg.inv(A + B @ Qinv)
    Qpinv   = Qp_n @ Qp_d
    
    return Qpinv

def ComputeOnTransversalPlane(baseray_pos,diffray_pos,baseray_dir,diffray_dir,surface_normal):

    
    
    # Transversal Plane basis vectors
    z = baseray_dir
    x = np.empty(baseray_pos.shape) # np.cross(z,surface_normal)
    y = np.empty(baseray_pos.shape) #np.cross(z,x)
    O = np.empty([3,3,baseray_pos.shape[-1]])
    
    for i in range(z.shape[-1]):
        x[:,i] = np.cross(z[:,i],surface_normal[:,i])
        y[:,i] = np.cross(x[:,i],z[:,i])
        O[:,:,i] = np.array([[x[0,i],y[0,i],z[0,i]],
                             [x[1,i],y[1,i],z[1,i]],
                             [x[2,i],y[2,i],z[2,i]]])
        
        
    
    # Shift differential ray to transversal plane
    rdiff = diffray_pos - baseray_pos
    
    # Shift differential dir to transversal plane
    # The second part of this eq is a vector projection of the diffray onto the z vector
    kdiff = diffray_dir - (np.sum(diffray_dir*z,axis=0))*z
    
    # Batch the dot products to get out of for loops
    dX = np.sum(rdiff*x,axis=0)
    dY = np.sum(rdiff*y,axis=0)
    dZ = np.sum(rdiff*z,axis=0) # should be zero, but they aren't, just small. Investigate potential bug
    
    dL = np.sum(kdiff*x,axis=0)
    dM = np.sum(kdiff*y,axis=0)
    dN = np.sum(kdiff*z,axis=0) # should be zero, but they aren't, just small.
    
    if (np.abs(dZ) >= 1e-10).any() or (np.abs(dN) >= 1e-10).any():
        print('Condition Violated, nonzero z components > 1e-')
        print(dZ)
        # print(dN)
        
    
    return dX,dY,dL,dM,O
    
    

def ComputeDifferentialFromRaybundles(raybundle0,raybundle1,raybundle2,raybundle3,raybundle4):

    # Parse the incoming raydata and finite difference
    # Below is the denomenator of the differential ray data
    xin0 = raybundle0.xData[0]
    yin0 = raybundle0.yData[0]
    lin0 = raybundle0.lData[0]
    min0 = raybundle0.mData[0]
    
    # The +Px ray
    dX0 = raybundle1.xData[0] - xin0
    
    # The +Py ray
    dY0 = raybundle2.yData[0] - yin0
    
    # The +Hx ray
    dL0 = raybundle3.lData[0] - lin0
    
    # The +Hy ray
    dM0 = raybundle4.mData[0] - min0

    # Parse the outgoing raydata, need all for a generally skew system
    # Below is the data we use to compute the numerator of the differential ray data
    xout0 = raybundle0.xData[-1]
    yout0 = raybundle0.yData[-1]
    zout0 = raybundle0.zData[-1]
    lout0 = raybundle0.lData[-1]
    mout0 = raybundle0.mData[-1]
    nout0 = raybundle0.nData[-1]
    
    
    xout1 = raybundle1.xData[-1]
    yout1 = raybundle1.yData[-1]
    zout1 = raybundle1.zData[-1]
    lout1 = raybundle1.lData[-1]
    mout1 = raybundle1.mData[-1]
    nout1 = raybundle1.nData[-1]

    xout2 = raybundle2.xData[-1]
    yout2 = raybundle2.yData[-1]
    zout2 = raybundle2.zData[-1]
    lout2 = raybundle2.lData[-1]
    mout2 = raybundle2.mData[-1]
    nout2 = raybundle2.nData[-1]
    
    xout3 = raybundle3.xData[-1]
    yout3 = raybundle3.yData[-1]
    zout3 = raybundle3.zData[-1]
    lout3 = raybundle3.lData[-1]
    mout3 = raybundle3.mData[-1]
    nout3 = raybundle3.nData[-1]

    xout4 = raybundle4.xData[-1]
    yout4 = raybundle4.yData[-1]
    zout4 = raybundle4.zData[-1]
    lout4 = raybundle4.lData[-1]
    mout4 = raybundle4.mData[-1]
    nout4 = raybundle4.nData[-1]
    
    # Call the finite difference evaluation on the transversal plane
    surface_normal = np.array([raybundle0.l2Data[-1],
                               raybundle0.m2Data[-1],
                               raybundle0.n2Data[-1]])
                               
    # Put these into rays to vectorize the dot product
    baseray_pos = np.array([xout0,yout0,zout0])
    diffray_pos_Px = np.array([xout1,yout1,zout1])
    diffray_pos_Py = np.array([xout2,yout2,zout2])
    diffray_pos_Hx = np.array([xout3,yout3,zout3])
    diffray_pos_Hy = np.array([xout4,yout4,zout4])
    
    baseray_dir = np.array([lout0,mout0,nout0])
    diffray_dir_Px = np.array([lout1,mout1,nout1])
    diffray_dir_Py = np.array([lout2,mout2,nout2])
    diffray_dir_Hx = np.array([lout3,mout3,nout3])
    diffray_dir_Hy = np.array([lout4,mout4,nout4])
    
    # First column of ray transfer matrix
    dX1,dY1,dL1,dM1,O = ComputeOnTransversalPlane(baseray_pos,diffray_pos_Px,baseray_dir,diffray_dir_Px,surface_normal)
    
    # Second column
    dX2,dY2,dL2,dM2,O = ComputeOnTransversalPlane(baseray_pos,diffray_pos_Py,baseray_dir,diffray_dir_Py,surface_normal)
    
    # Third column
    dX3,dY3,dL3,dM3,O = ComputeOnTransversalPlane(baseray_pos,diffray_pos_Hx,baseray_dir,diffray_dir_Hx,surface_normal)
    
    # Fourt column
    dX4,dY4,dL4,dM4,O = ComputeOnTransversalPlane(baseray_pos,diffray_pos_Hy,baseray_dir,diffray_dir_Hy,surface_normal)
    

    # Compute the differential ray transfer matrix from these data
    dMat = np.array([[dX1/dX0,dX2/dY0,dX3/dL0,dX4/dM0],
                     [dY1/dX0,dY2/dY0,dY3/dL0,dY4/dM0],
                     [dL1/dX0,dL2/dY0,dL3/dL0,dL4/dM0],
                     [dM1/dX0,dM2/dY0,dM3/dL0,dM4/dM0]])

    return dMat,O

@njit
def PropQParams(t_base,dMat,Qinv,x1,x2,y1,y2,k,opd):

    # Construct amplitude and phase
    Amplitude = np.empty(t_base.shape[-1],dtype='complex128')
    Phase = np.empty(t_base.shape[-1],dtype='complex128')

    for j in range(t_base.shape[-1]):
            
        # Sub-matrices
        A = dMat[0:2,0:2,j]
        B = dMat[0:2,2:4,j]
        C = dMat[2:4,0:2,j]
        D = dMat[2:4,2:4,j]
        
        # Q parameter propagation
        if np.abs(np.linalg.det(A + B @ Qinv)) == 0:
            qpinv = np.array([[1 + 0*1j,0*1j],[0*1j,1 + 0*1j]])
            # Turn off the beamlet
            Amplitude[j] = 0 # 1/np.sqrt(np.linalg.det(A + B @ Qinv))
            print('ERROR: cant propagate q parameter, turning off beamlet at pixel')
        else:
            
            qpinv = (C + D @ Qinv) @ np.linalg.inv(A + B @ Qinv)
            # Evaluate amplitude
            Amplitude[j] = 1/np.sqrt(np.linalg.det(A + B @ Qinv)) * 1e-10
            # Qpinv.append(qpinv)
            
        M = qpinv
        # Evaluate phasor 
        transversal = (-1j*k/2)*(x2[j]*M[0,0] + y2[j]*M[1,0])*x1[j] + (x2[j]*M[0,1] + y2[j]*M[1,1])*y1[j]
        opticalpath = (-1j*k)*opd + t_base[j]
        
        Phase[j] = transversal + opticalpath
            
    return Amplitude,Phase
    
def eval_gausfield_worku(base_rays,Px_rays,Py_rays,Hx_rays,Hy_rays,
                         wavelength,wo,detsize,npix,
                         dX0,dY0,dL0,dM0,
                         detector_normal=np.array([0,0,1])):
                         
    """
    This function is going to be a beast because
    1) I'm more interested in demoing the physics than writing efficient code
    2) I only just understand the method, so programming it in the way I understand is helpful
    3) If there aren't external function calls I can put the whole thing in a numba loop to make it easy
    
    this is INTERMEDIATE, and will be zoomy later
    """
    
    # 0) Init the Q parameter
    zr = np.pi*wo**2/wavelength
    qinv = 1/(1j*zr)
    Qinv = np.array([[qinv,0],[0,qinv]])
    Qpinv = [] # list of propagated Q parameters
    k = 2*np.pi/wavelength

    # 1) Define sensor R = <X,Y,Z>
    X = np.linspace(-detsize/2,detsize/2,npix)
    X,Y = np.meshgrid(X,X)
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = 0*X
    R = np.array([X,Y,Z])
    
    # 2) Grab Ray family Positions & Directions
    r_base = np.array([base_rays.xData[-1],
                       base_rays.yData[-1],
                       base_rays.zData[-1]])
                       
    r_Px = np.array([Px_rays.xData[-1],
                     Px_rays.yData[-1],
                     Px_rays.zData[-1]])             

    r_Py = np.array([Py_rays.xData[-1],
                     Py_rays.yData[-1],
                     Py_rays.zData[-1]])             

    r_Hx = np.array([Hx_rays.xData[-1],
                     Hx_rays.yData[-1],
                     Hx_rays.zData[-1]])             

    r_Hy = np.array([Hy_rays.xData[-1],
                     Hy_rays.yData[-1],
                     Hy_rays.zData[-1]])   
    
    k_base = np.array([base_rays.lData[-1],
                       base_rays.mData[-1],
                       base_rays.nData[-1]])

    k_base /= np.sqrt(k_base[0,:]**2 + k_base[1,:]**2 + k_base[2,:]**2 )
                       
    k_Px = np.array([Px_rays.lData[-1],
                     Px_rays.mData[-1],
                     Px_rays.nData[-1]])             

    k_Py = np.array([Py_rays.lData[-1],
                     Py_rays.mData[-1],
                     Py_rays.nData[-1]])             

    k_Hx = np.array([Hx_rays.lData[-1],
                     Hx_rays.mData[-1],
                     Hx_rays.nData[-1]])             

    k_Hy = np.array([Hy_rays.lData[-1],
                     Hy_rays.mData[-1],
                     Hy_rays.nData[-1]])

    # 3) Compute the distance along k rays need to go to intersect the transversal plane
    # Start by looping over where each beamlet needs to go
    
    # Where we put the base ray propagation vector to do dot products
    k_box = np.empty(R.shape)

    # Where we put the thickness the beamlets need to propagate
    t_box = np.empty(R.shape)

    # Where we put the detector pixel coordinate under analysis
    r_box = np.empty(R.shape)
    
    # npix x nbeamlets grid
    Phase = np.empty([R.shape[-1],k_base.shape[-1]],dtype='complex128')
    Amplitude = Phase
    print('eval phasor of shape = ',Phase.shape)

    # Loop over nbeamlets 
    for i in range(k_base.shape[-1]):
        
        # Make Vectors Multiplicable and compute distance (t) to transversal plane
        # Then update the position of the ray
        
        # Beamlet Coordinate System
        # Compute the beamlet transverse basis vectors
        z_beam = k_base[:,i]
        x_beam = np.cross(k_base[:,i],-detector_normal)
        y_beam = np.cross(k_base[:,i],x_beam)
        
        O = np.array([x_beam,y_beam,z_beam])
        
        # Base
        k_box[0,:] = k_base[0,i]
        k_box[1,:] = k_base[1,i]
        k_box[2,:] = k_base[2,i]
        # t_base = (np.sum(k_box*R,axis=0) - np.sum(k_base[:,i]*r_base[:,i],axis=0))/np.dot(k_base[:,i],k_base[:,i])
        t_base = np.sum(k_box*R,axis=0) - np.sum(z_beam*r_base[:,i],axis=0)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(t_base)

        # plt.figure()
        # plt.plot(t_base,'*')
        # plt.show()

        # plt.figure()
        # plt.title('ray coordinates at detector')
        # plt.plot(r_base[2,:],r_base[0,:],'*',label='yz')
        # plt.plot(r_base[2,:],r_base[1,:],'*',label='xz')
        # plt.legend()
        # plt.show()
        # # break point
        # O.append(None)

        # make the same size as the box of k vectors
        t_box[0,:] = t_base
        t_box[1,:] = t_base
        t_box[2,:] = t_base
        
        # add to initial position.
        r_box[0,:] = r_base[0,i]
        r_box[1,:] = r_base[1,i]
        r_box[2,:] = r_base[2,i]
        r_base_transversal = r_box + k_box*t_box # position on the transversal plane
        
        # Px
        k_box[0,:] = k_Px[0,i]
        k_box[1,:] = k_Px[1,i]
        k_box[2,:] = k_Px[2,i]
        # t_Px = (np.sum(k_box*R,axis=0) - np.sum(k_Px[:,i]*r_Px[:,i],axis=0))/np.dot(k_Px[:,i],k_Px[:,i])
        t_Px = np.sum(k_box*R,axis=0) - np.sum(k_Px[:,i]*r_Px[:,i],axis=0)

        t_box[0,:] = t_Px
        t_box[1,:] = t_Px
        t_box[2,:] = t_Px
        r_box[0,:] = r_Px[0,i]
        r_box[1,:] = r_Px[1,i]
        r_box[2,:] = r_Px[2,i]
        r_Px_transversal = r_box + k_box*t_box
        
        # Py
        k_box[0,:] = k_Py[0,i]
        k_box[1,:] = k_Py[1,i]
        k_box[2,:] = k_Py[2,i]
        # t_Py = (np.sum(k_box*R,axis=0) - np.sum(k_Py[:,i]*r_Py[:,i],axis=0))/np.dot(k_Py[:,i],k_Py[:,i])
        t_Py = np.sum(k_box*R,axis=0) - np.sum(k_Py[:,i]*r_Py[:,i],axis=0)

        t_box[0,:] = t_Py
        t_box[1,:] = t_Py
        t_box[2,:] = t_Py
        r_box[0,:] = r_Py[0,i]
        r_box[1,:] = r_Py[1,i]
        r_box[2,:] = r_Py[2,i]
        r_Py_transversal = r_box + k_box*t_box
        
        # Hx
        k_box[0,:] = k_Hx[0,i]
        k_box[1,:] = k_Hx[1,i]
        k_box[2,:] = k_Hx[2,i]
        # t_Hx = (np.sum(k_box*R,axis=0) - np.sum(k_Hx[:,i]*r_Hx[:,i],axis=0))/np.dot(k_Hx[:,i],k_Hx[:,i])
        t_Hx = np.sum(k_box*R,axis=0) - np.sum(k_Hx[:,i]*r_Hx[:,i],axis=0)

        t_box[0,:] = t_Hx
        t_box[1,:] = t_Hx
        t_box[2,:] = t_Hx
        r_box[0,:] = r_Hx[0,i]
        r_box[1,:] = r_Hx[1,i]
        r_box[2,:] = r_Hx[2,i]
        r_Hx_transversal = r_box + k_box*t_box
        
        # Hy
        k_box[0,:] = k_Hy[0,i]
        k_box[1,:] = k_Hy[1,i]
        k_box[2,:] = k_Hy[2,i]
        # t_Hy = (np.sum(k_box*R,axis=0) - np.sum(k_Hy[:,i]*r_Hy[:,i],axis=0))/np.dot(k_Hy[:,i],k_Hy[:,i])
        t_Hy = np.sum(k_box*R,axis=0) - np.sum(k_Hy[:,i]*r_Hy[:,i],axis=0)

        t_box[0,:] = t_Hy
        t_box[1,:] = t_Hy
        t_box[2,:] = t_Hy
        r_box[0,:] = r_Hy[0,i]
        r_box[1,:] = r_Hy[1,i]
        r_box[2,:] = r_Hy[2,i]
        r_Hy_transversal = r_box + k_box*t_box

        # plt.figure()
        # plt.plot(t_base,'*',label='base',alpha=0.5)
        # plt.plot(t_Px,'*',label='Px',alpha=0.5)
        # plt.plot(t_Py,'*',label='Py',alpha=0.5)
        # plt.plot(t_Hx,'*',label='Hx',alpha=0.5)
        # plt.plot(t_Hy,'*',label='Hy',alpha=0.5)
        # plt.legend()
        # plt.show()
        
        ## Now that all of the rays are at the transversal plane, we compute the ABCD matrix on them
        r_diff_Px = O @ r_Px_transversal - O @ r_base_transversal
        r_diff_Py = O @ r_Py_transversal - O @ r_base_transversal
        r_diff_Hx = O @ r_Hx_transversal - O @ r_base_transversal
        r_diff_Hy = O @ r_Hy_transversal - O @ r_base_transversal
        
        
        
        # Not sure if this is necessary any longer? Compute anyways
        # k_diff_Px = k_Px[:,i] - (np.sum(k_Px[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        # k_diff_Py = k_Py[:,i] - (np.sum(k_Py[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        # k_diff_Hx = k_Hx[:,i] - (np.sum(k_Hx[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        # k_diff_Hy = k_Hy[:,i] - (np.sum(k_Hy[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        
        k_diff_Px = O @ k_Px[:,i] - O @ k_base[:,i] # (np.sum(k_Px[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        k_diff_Py = O @ k_Py[:,i] - O @ k_base[:,i] #(np.sum(k_Py[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        k_diff_Hx = O @ k_Hx[:,i] - O @ k_base[:,i] #(np.sum(k_Hx[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        k_diff_Hy = O @ k_Hy[:,i] - O @ k_base[:,i] #(np.sum(k_Hy[:,i]*k_base[:,i],axis=0))*k_base[:,i]
        
        
        ## Project onto the local beamlet coordinate system
        
        # Pixels rotated into transversal plane basis
        r_base_on_transversal = O @ (R + r_base_transversal)
        r_Px_on_transversal = O @ (R - r_Px_transversal)
        r_Py_on_transversal = O @ (R - r_Py_transversal)
        r_Hx_on_transversal = O @ (R - r_Hx_transversal)
        r_Hy_on_transversal = O @ (R - r_Hy_transversal)

        
        # r_base_transversal is the position that intersects the plane defined by:
        # - the beamlet normal
        # - the detector pixel

        # Should be zero in the z dimension if were succesfully rotated onto the transversal plane, right?
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(r_base_on_transversal[0,:],r_base_on_transversal[1,:],r_base_on_transversal[2,:],label='base plane')
        # ax.scatter(r_Px_on_transversal[0,:],r_Px_on_transversal[1,:],r_Px_on_transversal[2,:],label='Px plane')
        # ax.scatter(r_Py_on_transversal[0,:],r_Py_on_transversal[1,:],r_Py_on_transversal[2,:],label='Py plane')
        # ax.scatter(r_Hx_on_transversal[0,:],r_Hx_on_transversal[1,:],r_Hx_on_transversal[2,:],label='Hx plane')
        # ax.scatter(r_Hy_on_transversal[0,:],r_Hy_on_transversal[1,:],r_Hy_on_transversal[2,:],label='Hy plane')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # # plt.zlabel('Z')
        # plt.show()

        x_box = np.empty(r_diff_Px.shape)
        y_box = np.empty(r_diff_Px.shape)
        z_box = np.empty(r_diff_Px.shape)
        
        x_box[0,:] = x_beam[0]
        x_box[1,:] = x_beam[1]
        x_box[2,:] = x_beam[2]
        
        y_box[0,:] = y_beam[0]
        y_box[1,:] = y_beam[1]
        y_box[2,:] = y_beam[2]
        
        z_box[0,:] = z_beam[0]
        z_box[1,:] = z_beam[1]
        z_box[2,:] = z_beam[2]
        
        # Differential Ray Parameters
        dX1 = np.sum(r_diff_Px*x_box,axis=0)
        dY1 = np.sum(r_diff_Px*y_box,axis=0)
        dZ1 = np.sum(r_diff_Px*z_box,axis=0) # should be zero
        dL1 = np.sum(k_diff_Px*x_beam,axis=0)*np.ones(dX1.shape)
        dM1 = np.sum(k_diff_Px*y_beam,axis=0)*np.ones(dX1.shape)
        dN1 = np.sum(k_diff_Px*z_beam,axis=0)*np.ones(dX1.shape)
        
        dX2 = np.sum(r_diff_Py*x_box,axis=0)
        dY2 = np.sum(r_diff_Py*y_box,axis=0)
        dZ2 = np.sum(r_diff_Py*z_box,axis=0) # should be zero
        dL2 = np.sum(k_diff_Py*x_beam,axis=0)*np.ones(dX1.shape)
        dM2 = np.sum(k_diff_Py*y_beam,axis=0)*np.ones(dX1.shape)
        dN2 = np.sum(k_diff_Py*z_beam,axis=0)*np.ones(dX1.shape)
        
        dX3 = np.sum(r_diff_Hx*x_box,axis=0)
        dY3 = np.sum(r_diff_Hx*y_box,axis=0)
        dZ3 = np.sum(r_diff_Hx*z_box,axis=0)
        dL3 = np.sum(k_diff_Hx*x_beam,axis=0)*np.ones(dX1.shape)
        dM3 = np.sum(k_diff_Hx*y_beam,axis=0)*np.ones(dX1.shape)
        dN3 = np.sum(k_diff_Hx*z_beam,axis=0)*np.ones(dX1.shape)
        
        dX4 = np.sum(r_diff_Hy*x_box,axis=0)
        dY4 = np.sum(r_diff_Hy*y_box,axis=0)
        dZ4 = np.sum(r_diff_Hy*z_box,axis=0)
        dL4 = np.sum(k_diff_Hy*x_beam,axis=0)*np.ones(dX1.shape)
        dM4 = np.sum(k_diff_Hy*y_beam,axis=0)*np.ones(dX1.shape)
        dN4 = np.sum(k_diff_Hy*z_beam,axis=0)*np.ones(dX1.shape)
        
        # plt.figure()
        # plt.title('Should be zero')
        # plt.plot(dZ1,label='dZ1')
        # plt.plot(dZ2,label='dZ2')
        # plt.plot(dZ3,label='dZ3')
        # plt.plot(dZ4,label='dZ4')

        # plt.plot(dN1,label='dN1')
        # plt.plot(dN2,label='dN2')
        # plt.plot(dN3,label='dN3')
        # plt.plot(dN4,label='dN4')
        # plt.legend()
        # plt.show()
        
        
        # Construct the Differential Ray Transfer Matrix
        # Should be a 4 x 4 x npix**2 matrix
        dMat = np.array([[dX1/dX0,dX2/dY0,dX3/dL0,dX4/dM0],
                         [dY1/dX0,dY2/dY0,dY3/dL0,dY4/dM0],
                         [dL1/dX0,dL2/dY0,dL3/dL0,dL4/dM0],
                         [dM1/dX0,dM2/dY0,dM3/dL0,dM4/dM0]],dtype='complex128')
        
        
        
        #print('Propagating Q')
        # propagate the Q parameter and assemble beamlet phases 
        
        Amplitude[:,i],Phase[:,i] = PropQParams(t_base,dMat,Qinv,
                                                r_base_on_transversal[0,:],r_base_on_transversal[0,:],
                                                r_base_on_transversal[1,:],r_base_on_transversal[1,:],
                                                k,base_rays.opd[-1][i])
                                                
        # print(Amplitude[:,i])                   
        # print(Phase[:,i]) 
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(r_on_transversal[0,:],r_on_transversal[1,:],r_on_transversal[2,:])
        # plt.show()
       
        # klist.append(None)
        # for j in range(t_base.shape[-1]):
            
            # # Sub-matrices
            # A = dMat[0:2,0:2,j]
            # B = dMat[0:2,2:4,j]
            # C = dMat[2:4,0:2,j]
            # D = dMat[2:4,2:4,j]
            
            # # Q parameter propagation
            # if np.abs(np.linalg.det(A + B @ Qinv)) == 0:
                # qpinv = np.identity(2)
                # # Turn off the beamlet
                # Amplitude[j,i] = 0# 1/np.sqrt(np.linalg.det(A + B @ Qinv))
            # else:
                
                # qpinv = (C + D @ Qinv) @ np.linalg.inv(A + B @ Qinv)
                # # Evaluate amplitude
                # Amplitude[j,i] = 1/np.sqrt(np.linalg.det(A + B @ Qinv))
                # # Qpinv.append(qpinv)
            
            # # Evaluate phasor 
            # transversal = (-1j*k/2)*Matmulvec(R[0,j],R[1,j],qpinv,R[0,j],R[1,j])
            # opticalpath = (-1j*k)*base_rays.opd[-1][i] + t_base[j]
            
            # Phase[j,i] = transversal + opticalpath
            
        
        # print('Beamlet #',i,' traced') 
    
    # do the field evaluation
    Phasor = ne.evaluate('exp(Phase)')
    Phasor *= Amplitude
    
    print('Evaluating Field')
    # Coherent sum along the beamlet axis
    Field = np.reshape(np.sum(Phasor,axis=-1),[npix,npix])
    print('Field Evaluation Completed')
    
    return Field
                         

            
        
        
        
        
        
        
    
    


def eval_gausfield(rays,sys,wavelength,wo,detsize,npix,O):

    # Ask the raybundle where the centroid is
    xToCen = rays.xData[-1]
    yToCen = rays.yData[-1]
    xCen = np.mean(xToCen)
    yCen = np.mean(yToCen)

    print('Centroid is x= ',xCen,' y = ',yCen)

    zr = np.pi*wo**2/wavelength
    Qinv = np.array([[-1/(1j*zr),0],[0,-1/(1j*zr)]])
    Q = np.linalg.inv(Qinv)
    k = 2*np.pi/wavelength

    # define detector axis u,v
    u = np.linspace(-detsize/2,detsize/2,npix)
    v = u
    u,v = np.meshgrid(u,u)
    u = np.ravel(u)
    v = np.ravel(v)
    
    # the box we put all the wavefront info in
    Dphase = np.empty([int(len(u)),rays.xData[0].shape[-1]],dtype='complex128')

    # replace with least-squares ampltidue fit later for arb amplitude
    amps = np.ones(rays.xData[0].shape[-1],dtype='complex128')
    for i in np.arange(0,rays.xData[0].shape[-1]):
        
        ray_original   = np.array([rays.xData[0][i],
                                   rays.yData[0][i],
                                   rays.zData[0][i]])
        ray_propagated = np.array([rays.xData[-1][i],
                                   rays.yData[-1][i],
                                   rays.zData[-1][i]])

        # A = sys[0:2,0:2,np.sqrt(rays.xData[0]**2 + rays.yData[0]**2) <= 0.1][:,:,0]
        # B = sys[0:2,2:4,np.sqrt(rays.xData[0]**2 + rays.yData[0]**2) <= 0.1][:,:,0]
        # C = sys[2:4,0:2,np.sqrt(rays.xData[0]**2 + rays.yData[0]**2) <= 0.1][:,:,0]
        # D = sys[2:4,2:4,np.sqrt(rays.xData[0]**2 + rays.yData[0]**2) <= 0.1][:,:,0]

        A = sys[0:2,0:2,i]
        B = sys[0:2,2:4,i]
        C = sys[2:4,0:2,i]
        D = sys[2:4,2:4,i]

        # print(A.shape)
        # print(B.shape)
        # print(C.shape)
        # print(D.shape)
        
        # Step 2 - Propagate Complex Beam Parameter
        Qp_n = (C + D @ Qinv)
        Qp_d = np.linalg.inv(A + B @ Qinv)
        Qpinv   = Qp_n @ Qp_d
        guoy_phase = -1j*ComputeGouyPhase(Qpinv)
        
        if np.linalg.det(A) <= 1e-10:
            orig_matrx = np.zeros([2,2])
        else:
            orig_matrx = np.linalg.inv(Q + np.linalg.inv(A) @ B)

        cros_matrx = np.linalg.inv(A @ Q + B)

        
        # Decenter Parameter
        uo = ray_original[0] #+ u*0
        vo = ray_original[1] #+ v*0

        lo = rays.opd[-1][i] # This is the parameter that's been missing the whole time, need to compute the actual OPD if it exits

        # Shift detector by final beamlet position. Analogous to moving to central ray position
        up = u-ray_propagated[0] + xCen 
        vp = v-ray_propagated[1] + yCen
        
        # Now the ray needs to be projected onto the transversal plane, how do we do this?
        coords_to_rotate = np.array([up,vp,0*up])
        rotated = np.transpose(O[:,:,i]) @ coords_to_rotate
        up = rotated[0,:]
        vp = rotated[1,:]
        
        # print(rotated[2,:])
        
        # The orthogonal transformation
        #Qpinv = np.array([[Qpinv[0,0],Qpinv[0,1],0],
        #                  [Qpinv[1,0],Qpinv[1,1],0],
        #                  [0,0,1]])
        #ray_to_prop = 
        

        # All the beamlet phases
        # kernel = np.transpose(np.array([up,vp,0*up])) @ O[:,:,i] @ Qpinv @ np.transpose(O[:,:,i]) @ np.array([up,vp,0*up])
        # print(kernel.shape)
        # kernel = np.ravel(kernel)
        
        tran_phase = -1j*k/2*Matmulvec(up,vp,Qpinv,up,vp)
        long_phase = 1j*k*lo
        # orig_phase = -1j*k/2*np.transpose(np.array([uo,vo])) @ orig_matrx @ np.array([uo,vo]) # Matmulvec(uo,vo,orig_matrx,uo,vo)*0
        # cros_phase = 1j*k*np.transpose(np.array([uo,vo])) @ orig_matrx @ np.array([up,vp]) # Matmulvec(uo,vo,cros_matrx,up,vp)*0
        orig_phase = -1j*k/2*Matmulvec(uo,vo,orig_matrx,uo,vo)
        cros_phase = 1j*k*Matmulvec(uo,vo,cros_matrx,up,vp)
        
        # The beamlet amplitude
        amps[i] *= 1/np.sqrt(np.linalg.det(A + B @ Qinv))

        # load the phase arrays for Worku's equation
        Dphase[:,i] = guoy_phase + tran_phase + long_phase 
        
        # load phase arrays from Cai and Lin
        Dphase[:,i] += orig_phase + cros_phase

        # Not computationally efficient, but this allows us to filter the bad values
        #if lo == 0:
        #    Dphase[:,i] = 0
        #    amps[i] = 0
            
    

    # Now we evaluate the phasor and sum the beamlets
    # use numexpr so it doesn't take forever
    # are the amplitude axes aligned to the sum?
    
    #Efield = np.sum(amps*ne.evaluate('exp(Dphase)'),axis=1)
    Efield = np.sum(amps*ne.evaluate('exp(Dphase)'),axis=1)

    return Efield
