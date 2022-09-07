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
            Amplitude[j] = 1/np.sqrt(np.linalg.det(A + B @ Qinv)) #* 1e-10
            # Qpinv.append(qpinv)
            
        M = qpinv
        # Evaluate phasor 
        transversal = (-1j*k/2)*((x2[j]*M[0,0] + y2[j]*M[1,0])*x1[j] + (x2[j]*M[0,1] + y2[j]*M[1,1])*y1[j])


        opticalpath = (-1j*k)*(opd + t_base[j])
        
        Phase[j] = transversal + opticalpath
            
    return Amplitude,Phase

def EvalGausfieldWorku(base_rays,Px_rays,Py_rays,Hx_rays,Hy_rays,
                         wavelength,wo,detsize,npix,
                         dX0,dY0,dL0,dM0,
                         detector_normal=np.array([0,0,1])):

    """Second-generation function, this is after some digesting
    """

     # 0) Init the Q parameter
    zr = np.pi*wo**2/wavelength
    qinv = 1/(1j*zr)
    Qinv = np.array([[qinv,0],[0,qinv]])
    Qpinv = [] # list of propagated Q parameters
    k = 2*np.pi/wavelength

    # 1) Define sensor R = <X,Y,Z>
    # Consider offsetting by the centroid of base_rays
    X = np.linspace(-detsize/2,detsize/2,npix)
    X,Y = np.meshgrid(X,X)
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = 0*X
    R = np.array([X,Y,Z])

    # Add a displacement to the detector based on the r_base centroid
    centroid = np.array([np.mean(base_rays.xData[-1]),np.mean(base_rays.yData[-1]),np.mean(base_rays.zData[-1])])
    print('Image Centroid @ ',centroid)
    R[0] += centroid[0]
    R[1] += centroid[1]
    R[2] += centroid[2]

    
    # 2) Grab Ray family Positions & Directions
    r_base = np.array([[base_rays.xData[-1]],
                       [base_rays.yData[-1]],
                       [base_rays.zData[-1]]])
                       
    r_Px = np.array([[Px_rays.xData[-1]],
                     [Px_rays.yData[-1]],
                     [Px_rays.zData[-1]]])             

    r_Py = np.array([[Py_rays.xData[-1]],
                     [Py_rays.yData[-1]],
                     [Py_rays.zData[-1]]])             

    r_Hx = np.array([[Hx_rays.xData[-1]],
                     [Hx_rays.yData[-1]],
                     [Hx_rays.zData[-1]]])             

    r_Hy = np.array([[Hy_rays.xData[-1]],
                     [Hy_rays.yData[-1]],
                     [Hy_rays.zData[-1]]])   
    
    k_base = np.array([[base_rays.lData[-1]],
                       [base_rays.mData[-1]],
                       [base_rays.nData[-1]]])
                       
    k_Px = np.array([[Px_rays.lData[-1]],
                     [Px_rays.mData[-1]],
                     [Px_rays.nData[-1]]])             

    k_Py = np.array([[Py_rays.lData[-1]],
                     [Py_rays.mData[-1]],
                     [Py_rays.nData[-1]]])             

    k_Hx = np.array([[Hx_rays.lData[-1]],
                     [Hx_rays.mData[-1]],
                     [Hx_rays.nData[-1]]])             

    k_Hy = np.array([[Hy_rays.lData[-1]],
                     [Hy_rays.mData[-1]],
                     [Hy_rays.nData[-1]]])

    # npix x nbeamlets grid
    Phase = np.empty([R.shape[-1],k_base.shape[-1]],dtype='complex128')
    Amplitude = Phase
    print('eval phasor of shape = ',Phase.shape)

    # Loop over nbeamlets 
    for i in range(k_base.shape[-1]):

        ## Compute ray differentials on transvese plane
        # This function has two "modes", where if a differential input (dR, dK) is not supplied,
        # it returns the transversal plane information instead. First we will use it to return the differential info

        # the dPx differential ray
        A,_,C,_ = EvalDifferentialOnTransversal(r_base[:,:,i],k_base[:,:,i],r_Px[:,:,i],k_Px[:,:,i],R,dR=dX0,dK=0)
        Axx = A[0]
        Ayx = A[1]
        Cxx = C[0] * np.ones(Axx.shape)
        Cyx = C[1] * np.ones(Axx.shape)

        # the dPy differential ray
        A,_,C,_ = EvalDifferentialOnTransversal(r_base[:,:,i],k_base[:,:,i],r_Py[:,:,i],k_Py[:,:,i],R,dR=dY0,dK=0)
        Axy = A[0]
        Ayy = A[1]
        Cxy = C[0] * np.ones(Axx.shape)
        Cyy = C[1] * np.ones(Axx.shape)

        # the dHx differential ray
        _,B,_,D = EvalDifferentialOnTransversal(r_base[:,:,i],k_base[:,:,i],r_Hx[:,:,i],k_Hx[:,:,i],R,dR=0,dK=dL0)
        Bxx = B[0]
        Byx = B[1]
        Dxx = D[0] * np.ones(Axx.shape)
        Dyx = D[1] * np.ones(Axx.shape)

        # the dHy differential ray
        _,B,_,D = EvalDifferentialOnTransversal(r_base[:,:,i],k_base[:,:,i],r_Hy[:,:,i],k_Hy[:,:,i],R,dR=0,dK=dM0)
        Bxy = B[0]
        Byy = B[1]
        Dxy = D[0] * np.ones(Axx.shape)
        Dyy = D[1] * np.ones(Axx.shape)

        # print(Axx.shape)
        # print(Axy.shape)
        # print(Ayx.shape)
        # print(Ayy.shape)

        # print(Bxx.shape)
        # print(Bxy.shape)
        # print(Byx.shape)
        # print(Byy.shape)

        # print(Cxx.shape)
        # print(Cxy.shape)
        # print(Cyx.shape)
        # print(Cyy.shape)

        # print(Dxx.shape)
        # print(Dxy.shape)
        # print(Dyx.shape)
        # print(Dyy.shape)

        # Dyy.append(False)

        dMat = np.array([[Axx,Axy,Bxx,Bxy],
                         [Ayx,Ayy,Byx,Byy],
                         [Cxx,Cxy,Dxx,Dxy],
                         [Cyx,Cyy,Dyx,Dyy]],dtype='complex128')
        
        # grab the detector coordinates with the same function
        # supplying the base to the differential only means that the derivative is zero, so it doesn't affect the computation
        # BUT IT IS SLOWER THAN IT NEEDS TO BE
        r_detector,t_base = EvalDifferentialOnTransversal(r_base[:,:,i],k_base[:,:,i],r_base[:,:,i],k_base[:,:,i],R)
        
        #print('Propagating Q')
        # propagate the Q parameter and assemble beamlet phases 
        
        Amplitude[:,i],Phase[:,i] = PropQParams(t_base,dMat,Qinv,
                                                r_detector[0],r_detector[0],
                                                r_detector[1],r_detector[1],
                                                k,base_rays.opd[-1][i])

    # do the field evaluation
    Phasor = ne.evaluate('exp(Phase)')
    Phasor *= Amplitude
    
    print('Evaluating Field')
    # Coherent sum along the beamlet axis
    Field = np.reshape(np.sum(Phasor,axis=-1),[npix,npix])
    print('Field Evaluation Completed')
    
    return Field

        

def EvalDifferentialOnTransversal(R_base,K_base,R_diff,K_diff,R_detector,dR=0,dK=0,detector_normal=np.array([0.,0.,1.])):

    """ computes the elements of the ABCD matrix, this should be called 4 times, each time it returns 4 values of the ABCD matrix 
    """

    # compute basis vectors for transversal plane
    # negate normal to keep right-handed coordinates
    # print(K_base,' is shape ',K_base.shape)
    # print(detector_normal,' is shape ',detector_normal.shape)
    # xloc.append(1)
    zloc = K_base[:,0]
    xloc = np.cross(zloc,-detector_normal)
    xloc /= np.linalg.norm(xloc)
    yloc = np.cross(zloc,xloc)

    # Orthogonal transformation matrix
    Oin = np.array([[xloc[0],yloc[0],zloc[0]],
                    [xloc[1],yloc[1],zloc[1]],
                    [xloc[2],yloc[2],zloc[2]]])

    Oinv = np.transpose(Oin)

    # find thickness delta to propagate r_base and r_diff to
    # R_detector @ zloc is just a clever way of broadcasting a dot product, dimensions must be compatible
    # so we take the transpose. All other values should be scalar
    # delta_base,diff should be N x 1 arrays at the output

    delta_base = (R_detector.transpose() @ zloc - np.dot(zloc,R_base))
    delta_diff = (R_detector.transpose() @ zloc - np.dot(zloc,R_diff))/(np.dot(zloc,K_diff))

    # print('z shape ',zloc.shape)
    # print('K_diff shape ',K_diff.shape)
    # print('R_base shape ',R_base.shape)
    # print('R_detector shape ',R_detector.shape)
    # print('delta_base shape ',delta_base.shape)
    # 1 @ np.ones([20,False])
    # update ray position for each pixel
    # delta's also need to be compatible now, so we do another transpose
    # As long as the axis are aligned, we can just add R_base,diff succesfully
    R_base = R_base + K_base * delta_base # the transversal plane origin
    R_diff = R_diff + K_diff * delta_diff

    # rotate into transversal plane with Oinv
    # dimensions should be 3 x N, which are compatible with Oinv
    r_base = Oinv @ R_base 
    k_base = Oinv @ K_base

    r_diff = Oinv @ R_diff 
    k_diff = Oinv @ K_diff

    # compute finite differences, some of these will blow up
    if dR != 0:

        drdR = (r_diff - r_base)/dR # A matrix column
        dkdR = (k_diff - k_base)/dR # C matrix column
        drdK = None
        dkdK = None

        return drdR,drdK,dkdR,dkdK

    elif dK != 0:
        
        drdR = None
        dkdR = None
        drdK = (r_diff - r_base)/dK # B matrix column
        dkdK = (k_diff - k_base)/dK # D matrix column

        return drdR,drdK,dkdR,dkdK

    else:

        # return transformed detector pixels
        r_detector = Oinv @ (R_detector - R_base)

        return r_detector,delta_base

    
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
        t_Px = (np.sum(k_box*R,axis=0) - np.sum(k_Px[:,i]*r_Px[:,i],axis=0))#/(np.dot(z_beam,k_Px))

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
        r_base_on_transversal = O @ (R - r_base_transversal)
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
