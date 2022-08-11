import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne

def Matmulvec(x2,y2,M,x1,y1):

    return (x2*M[0,0] + y2*M[1,0])*x1 + (x2*M[0,1] + y2*M[1,1])*y1

def ComputeGouyPhase(Q):


    eigvals = np.linalg.eigvals(Q)
    q1,q2 = eigvals[0],eigvals[1]

    gouy = .5*(np.arctan(np.real(q1)/np.imag(q1)) + np.arctan(np.real(q2)/np.imag(q2)))

    return gouy

def ComputeOnTransversalPlane(baseray_pos,diffray_pos,baseray_dir,diffray_dir,surface_normal):
    
    # Transversal Plane basis vectors
    z = baseray_dir
    x = np.empty(baseray_pos.shape) # np.cross(z,surface_normal)
    y = np.empty(baseray_pos.shape) #np.cross(z,x)
    O = np.empty([3,3,baseray_pos.shape[-1]])
    
    for i in range(z.shape[-1]):
        x[:,i] = np.cross(z[:,i],-surface_normal[:,i])
        y[:,i] = np.cross(x[:,i],z[:,i])
        O[:,:,i] = np.array([[x[0,i],y[0,i],z[0,i]],
                             [x[1,i],y[1,i],z[1,i]],
                             [x[2,i],y[2,i],z[2,i]]])
        
        
    
    # Shift differential ray to transversal plane
    rdiff = diffray_pos - baseray_pos
    
    # Shift differential dir to transversal plane
    # The second part of this eq is a vector projection of the diffray onto the z vector
    kdiff = diffray_dir - (np.sum(diffray_dir*z,axis=0))*z
    
    dX = np.sum(rdiff*x,axis=0)
    dY = np.sum(rdiff*y,axis=0)
    dZ = np.sum(rdiff*z,axis=0) # should be zero
    
    dL = np.sum(kdiff*x,axis=0)
    dM = np.sum(kdiff*y,axis=0)
    dN = np.sum(kdiff*z,axis=0) # should be zero
    
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
    Efield = np.sum(amps*np.exp(Dphase),axis=1)

    return Efield
