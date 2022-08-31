# poke core functions
# This is the meat of the PRT calculation
# The raytracer already works, why not add gaussian beamlets?

# dependencies
import numpy as np
# import poke.thinfilms_prysm as tf
import poke.thinfilms as tf

# Step 1) Compute Fresnel Coefficients
def FresnelCoefficients(aoi,n1,n2,mode='reflection'):

    if type(n2) == list:
        if len(n2) == 2:

            if mode == 'reflection':

                n_film = n2[0][0]
                d_film = n2[0][1]
                n_sub  = n2[1][0]

                rs = HartenTwoLayerFilm(aoi,n_film,d_film,n_sub,'s')
                rp = HartenTwoLayerFilm(aoi,n_film,d_film,n_sub,'p')
                fs = rs
                fp = rp

        else:

            print('not valid 2 layer')
            aoi.append(True) # just breaks the code

    else:
        # ratio of refractive indices
        n = n2/n1

        if mode == 'reflection':

            rs = (np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
            rp = (n**2 * np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
            fs = rs
            fp = rp

        elif mode == 'transmission':

            ts = (2*np.cos(aoi))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
            tp = (2*n*np.cos(aoi))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
            fs = ts
            fp = tp

        # elif mode == 'both':

        #     ts = (2*np.cos(aoi))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        #     tp = (2*n*np.cos(aoi))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        #     rs = (np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        #     rp = (n**2 * np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

    return fs,fp

def HartenTwoLayerFilm(aoi,n_film,d_film,n_sub,polarization):
    # transform to the angle in the film, assume vacuum ambient
    aor_film = np.arcsin(np.sin(aoi)/n_film)
    aor_sub = np.arcsin(n_film*np.sin(aor_film)/n_sub) # <- does this work for complex n_sub?

    # Compute the beta value in the film
    df = 2*np.pi/600e-9 * d_film * n_film * np.cos(aor_film)

    if polarization == 'p':
        nm = np.cos(aoi) # medium 
        nf = n_film * np.cos(aor_film) # film 
        nb = n_sub * np.cos(aor_sub)# substrate 

    elif polarization == 's':
        nm = 1/np.cos(aoi) # medium 
        nf = n_film / np.cos(aor_film) # film 
        nb = n_sub / np.cos(aor_sub)# substrate 

    Em = np.cos(df) + 1j*(nb/nf)*np.sin(df)
    Hm = nb*np.cos(df) + 1j*nf*np.sin(df)

    rtot = (nm*Em - Hm)/(nm*Em + Hm)

    return rtot
        
        
# Step 2) Construct Orthogonal Transfer Matrices
def ConstructOrthogonalTransferMatrices(kin,kout,normal):
    
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
    # sout /= np.linalg.norm(sout) # normalize the s-vector
    pout = np.cross(kout,sout)
    pout /= np.linalg.norm(pout)
    Oout = np.transpose(np.array([sout,pout,kout]))

    return Oinv,Oout

# Step 3) Create Polarization Ray Trace matrix
def ConstructPRTMatrix(kin,kout,normal,aoi,n1,n2,wavelength,mode='reflection',recipe=None):
    normal = -normal

    # Compute the Fresnel coefficients for either transmission OR reflection
    if recipe == None:
        fs,fp = FresnelCoefficients(aoi,n1,n2,mode=mode)
    else:
        # prysm likes films in degress, wavelength in microns, thickness in microns
        # rs,ts = tf.multilayer_stack_rt(recipe, wavelength*1e6, 's', aoi=aoi*180/np.pi,assume_vac_ambient=True)
        # rp,tp = tf.multilayer_stack_rt(recipe, wavelength*1e6, 'p', aoi=aoi*180/np.pi,assume_vac_ambient=True)

        rs,ts,rp,tp = tf.ComputeThinFilmCoeffsCLY(recipe,aoi,wavelength)

        # is S conserved?
        # print('s test, should be unity')
        # print(np.abs(rs)**2 + np.abs(ts)**2)

        # print('p test, should be unity')
        # print(np.abs(rp)**2 + np.abs(tp)**2)

        # break point
        # fs.append(normal)
        
        if mode == 'reflection':
            fs = rs
            fp = rp
        if mode == 'transmission':
            fs = ts
            fp = tp

    # Compute the orthogonal transfer matrices
    Oinv,Oout = ConstructOrthogonalTransferMatrices(kin,kout,normal)

    # Compute the Jones matrix
    J = np.array([[fs,0,0],[0,fp,0],[0,0,1]])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])

    # Compute the Polarization Ray Tracing Matrix
    # Pmat = np.matmul(Oout,np.matmul(J,Oinv))
    Pmat = Oout @ J @ Oinv
    Omat = Oout @ B @ Oinv # The parallel transport matrix, return when ready to implement
    # print('P shape = ',Pmat.shape)
    # print('Pmat')
    # print(Pmat)
    # print('J shape = ',J.shape)
    # print('Oinv shape = ',Oinv.shape)
    # print('Oout shape = ',Oout.shape)

    # This returns the polarization ray tracing matrix but I'm not 100% sure its in the coordinate system of the Jones Pupil
    return Pmat,J

def GlobalToLocalCoordinates(Pmat,kin,k,a=[0,1,0],exit_x=np.array([-1.,0.,0.])):

    # Double Pole Coordinate System, requires a rotation about an axis
    # Wikipedia article seems to disagree with CLY Example 11.4

    # Okay so let's think about this from the perspective of how we build up an orthogonal transformation
    # Need kin, kout, normal
    # for arb ray bundle kin = kout = normal

    # Default entrance pupil for astronomical telescopes in Zemax
    xin = np.array([1.,0.,0.])#np.cross(kin,np.array([0,0,1]))
    xin /= np.linalg.norm(xin)
    yin = np.cross(kin,xin)
    yin /= np.linalg.norm(yin)
    O_e = np.array([[xin[0],yin[0],kin[0]],
                    [xin[1],yin[1],kin[1]],
                    [xin[2],yin[2],kin[2]]])

    # O_e = np.identity(3)
    # Compute Exit Pupil Basis Vectors
    # For arbitrary k each ray will have it's own pair of basis vectors
    # Get Exit Pupil Basis Vectors
    # th = -np.arccos(np.dot(k,a))
    r = np.cross(k,a)
    r /= np.linalg.norm(r)
    th = -vectorAngle(k,a)
    R = rotation3D(th,r)

    # Local basis vectors
    xout = exit_x
    yout = np.cross(a,xout)
    yout /= np.linalg.norm(yout)
    x = R @ xout
    x /= np.linalg.norm(x)
    y = R @ yout
    y /= np.linalg.norm(y)

    # x = np.array([1-k[0]**2/(1+k[1]),
    #               -k[0],
    #               -k[0]*k[2]/(1+k[1])])
    # y = np.array([k[0]*k[2]/(1+k[1]),
    #               k[2],
    #               k[2]**2/(1+k[1]) -1])

    # xout = exit_x #np.cross(a,k)
    # xout /= np.linalg.norm(xout)
    # yout = np.cross(a,xout)
    # yout /= np.linalg.norm(xout)
    # x = R @ xout
    # y = R @ yout

    O_x = np.array([[x[0],y[0],k[0]],
                    [x[1],y[1],k[1]],
                    [x[2],y[2],k[2]]])

    J = np.linalg.inv(O_x) @ Pmat @ O_e

    return J

def JonesToMueller(Jones):

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,1j,-1j,0]])

    U *= np.sqrt(1/2)

    M = U @ (np.kron(np.conj(Jones),Jones)) @ np.linalg.inv(U)

    return M

def MuellerToJones(M):

    "CLY Eq. 6.112"
    "Untested"

    print('This operation looses global phase')

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

def ComputeDRFromJones(J):

    from scipy.linalg import polar
    from numpy.linalg import eig,svd

    evals,evecs = eig(J) # will give us the rotations to quiver
    W,D,Vh = svd(J) # gives the diattenuation, retardance
    diattenuation = (np.max(D)**2 - np.min(D)**2)/(np.max(D)**2 + np.min(D)**2) # CLY 5.102
    U,P = polar(J)
    # U = W @ np.linalg.inv(Vh)
    uval,uvec = eig(U)
    retardance = np.abs(np.angle(uval[0])-np.angle(uval[1])) # CLY 5.81

    return evecs,diattenuation,retardance

def ComputeDRFromPRT(P):
    
    "Yun et al"
    
    from scipy.linalg import svd,eig
    #print(P)
    W,D,Vh = svd(P)
    eigvals,eigvecs = eig(P)
    
    print(np.abs(eigvals))
    
    
    #print(W)
    #print(D)
    #print(Vh)
    
    # singular values given in descending order
    L1 = D[1]
    L2 = D[2]
    #print(L1)
    #print(L2)
    diattenuation = (np.abs(L1)**2 - np.abs(L2)**2)/(np.abs(L1)**2 + np.abs(L2)**2)
    retardance = np.angle(eigvals[2]) - np.angle(eigvals[1])
    
    return diattenuation,retardance#,Vh[1,:],Vh[2,:]
    
def ComputeDRFromAOI(aoi,n1,n2,mode='reflection'):

    fs,fp = FresnelCoefficients(aoi,n1,n2,mode=mode)
    
    diattenuation = (np.abs(fs)**2 - np.abs(fp)**2)/(np.abs(fs)**2 + np.abs(fp)**2)
    retardance = np.angle(fs) - np.angle(fp)
    
    return diattenuation,retardance
    
    
def PauliSpinMatrix(i):

    if i == 0:
    
        return np.array([[1,0],[0,1]])
        
    if i == 1:
    
        return np.array([[1,0],[0,-1]])
        
    if i == 2:
        return np.array([[0,1],[1,0]])
        
    if i == 3:
    
        return np.array([[0,-1j],[1j,0]])
        
def ComputePauliCoefficients(J):

    # Isotropic Plate
    c0 = np.trace(J @ PauliSpinMatrix(0))
    c1 = np.trace(J @ PauliSpinMatrix(1))
    c2 = np.trace(J @ PauliSpinMatrix(2))
    c3 = np.trace(J @ PauliSpinMatrix(3))
    
    return c0,c1,c2,c3
    
def DiattenuationAndRetardancdFromPauli(J):
    
    c0,c1,c2,c3 = ComputePauliCoefficients(J)
    c1 /= c0
    c2 /= c0
    c3 /= c0
    
    amp = np.abs(c0)
    phase = np.angle(c0)
    
    linear_diattenuation_hv = np.real(c1)
    linear_retardance_hv = np.imag(c1)
    
    linear_diattenuation_45 = np.real(c2)
    linear_retardance_45 = np.imag(c2)
    
    circular_diattenuation = np.real(c3)
    circular_retardance = np.imag(c3)
    
    diattenuation = [linear_diattenuation_hv,linear_diattenuation_45,circular_diattenuation]
    retardance = [linear_retardance_hv,linear_retardance_45,circular_retardance]
    
    return amp,phase,diattenuation,retardance
    
    
    
    
    
    


    
    

"Vector Operations from Quinn Jarecki"
import numpy as np
import cmath as cm

def rotation3D(angle,axis):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[(1-c)*axis[0]**2 + c, (1-c)*axis[0]*axis[1] - s*axis[2], (1-c)*axis[0]*axis[2] + s*axis[1]],
                    [(1-c)*axis[1]*axis[0] + s*axis[2], (1-c)*axis[1]**2 + c, (1-c)*axis[1]*axis[2] - s*axis[0]],
                    [(1-c)*axis[2]*axis[0] - s*axis[1], (1-c)*axis[1]*axis[2] + s*axis[0], (1-c)*axis[2]**2 + c]])
    return mat

def vectorAngle(u,v):
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    if u@v<0:
        return np.pi - 2*np.arcsin(np.linalg.norm(-v-u)/2)
    else:
        return 2*np.arcsin(np.linalg.norm(v-u)/2)
