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

def ComputeDifferentialFromRaybundles(raybundle0,raybundle1,raybundle2,raybundle3,raybundle4):

    # Parse the incoming raydata and finite difference
    xin0 = raybundle0.xData[0]
    yin0 = raybundle0.yData[0]
    uin0 = raybundle0.lData[0]/raybundle0.nData[0]
    vin0 = raybundle0.mData[0]/raybundle0.nData[0]
    
    xin1 = raybundle1.xData[0] - xin0
    yin1 = raybundle1.yData[0] - yin0
    uin1 = raybundle1.lData[0]/raybundle1.nData[0] - uin0
    vin1 = raybundle1.mData[0]/raybundle1.nData[0] - vin0

    xin2 = raybundle2.xData[0] - xin0
    yin2 = raybundle2.yData[0] - yin0
    uin2 = raybundle2.lData[0]/raybundle2.nData[0] - uin0
    vin2 = raybundle2.mData[0]/raybundle2.nData[0] - vin0
    
    xin3 = raybundle3.xData[0] - xin0
    yin3 = raybundle3.yData[0] - yin0
    uin3 = raybundle3.lData[0]/raybundle3.nData[0] - uin0
    vin3 = raybundle3.mData[0]/raybundle3.nData[0] - vin0

    xin4 = raybundle4.xData[0] - xin0
    yin4 = raybundle4.yData[0] - yin0
    uin4 = raybundle4.lData[0]/raybundle4.nData[0] - uin0
    vin4 = raybundle4.mData[0]/raybundle4.nData[0] - vin0

    # Parse the outgoing raydata
    xout0 = raybundle0.xData[-1]
    yout0 = raybundle0.yData[-1]
    uout0 = raybundle0.lData[-1]/raybundle0.nData[-1]
    vout0 = raybundle0.mData[-1]/raybundle0.nData[-1]
    
    xout1 = raybundle1.xData[-1] - xout0
    yout1 = raybundle1.yData[-1] - yout0
    uout1 = raybundle1.lData[-1]/raybundle1.nData[-1] - uout0
    vout1 = raybundle1.mData[-1]/raybundle1.nData[-1] - vout0

    xout2 = raybundle2.xData[-1] - xout0
    yout2 = raybundle2.yData[-1] - yout0
    uout2 = raybundle2.lData[-1]/raybundle2.nData[-1] - uout0
    vout2 = raybundle2.mData[-1]/raybundle2.nData[-1] - vout0
    
    xout3 = raybundle3.xData[-1] - xout0
    yout3 = raybundle3.yData[-1] - yout0
    uout3 = raybundle3.lData[-1]/raybundle3.nData[-1] - uout0
    vout3 = raybundle3.mData[-1]/raybundle3.nData[-1] - vout0

    xout4 = raybundle4.xData[-1] - xout0
    yout4 = raybundle4.yData[-1] - yout0
    uout4 = raybundle4.lData[-1]/raybundle4.nData[-1] - uout0
    vout4 = raybundle4.mData[-1]/raybundle4.nData[-1] - vout0

    # Compute the differential ray transfer matrix from these data
    dMat = np.array([[xout1/xin1,xout2/yin2,xout3/uin3,xout4/vin4],
                     [yout1/xin1,yout2/yin2,yout3/uin3,yout4/vin4],
                     [uout1/xin1,uout2/yin2,uout3/uin3,uout4/vin4],
                     [vout1/xin1,vout2/yin2,vout3/uin3,vout4/vin4]])

    return dMat




def eval_gausfield(rays,sys,wavelength,wo,detsize,npix):

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
    u,v = np.meshgrid(u,u)
    u = np.ravel(u)
    v = np.ravel(v)
    # the box we put all the wavefront info in
    Dphase = np.empty([len(u),rays.xData[0].shape[-1]],dtype='complex128')

    # replace with least-squares ampltidue fit later for arb amplitude
    amps = np.ones(rays.xData[0].shape[-1],dtype='complex128')
    print(rays)
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
        
        if np.linalg.det(A) <= 1e-10:
            orig_matrx = np.zeros([2,2])
        else:
            orig_matrx = np.linalg.inv(Q + np.linalg.inv(A) @ B)

        cros_matrx = np.linalg.inv(A @ Q + B)

        
        # Decenter Parameter
        uo = ray_original[0]
        vo = ray_original[1]

        lo = rays.opd[-1][i] # This is the parameter that's been missing the whole time, need to compute the actual OPD if it exits

        # Shift detector by final beamlet position
        up = u-ray_propagated[0] + xCen 
        vp = v-ray_propagated[1] + 1.3*yCen

        # All the beamlet phases
        guoy_phase = -1j*ComputeGouyPhase(Qpinv)
        tran_phase = -1j*k/2*Matmulvec(up,vp,Qpinv,up,vp)
        long_phase = 1j*k*lo
        orig_phase = -1j*k/2*Matmulvec(uo,vo,orig_matrx,uo,vo)
        cros_phase = 1j*k*Matmulvec(uo,vo,cros_matrx,up,vp)

        # The beamlet amplitude
        amps[i] *= 1/np.sqrt(np.linalg.det(A + B @ Qinv))

        # laod the phase arrays
        Dphase[:,i] = guoy_phase + tran_phase + long_phase + orig_phase + cros_phase

        # Not computationally efficient, but this allows us to filter the bad values
        if lo == 0:
            Dphase[:,i] = 0
            amps[i] = 0

    # Now we evaluate the phasor and sum the beamlets
    # use numexpr so it doesn't take forever
    # are the amplitude axes aligned to the sum?

    Efield = np.sum(amps*ne.evaluate('exp(Dphase)'),axis=1)

    return Efield
