# poke core functions
# This is the meat of the PRT calculation
# The raytracer already works, why not add gaussian beamlets?

# dependencies
import numpy as np

# Step 1) Compute Fresnel Coefficients
def FresnelCoefficients(aoi,n1,n2,mode='reflection'):

    # ratio of refractive indices
    n = n2/n1

    if mode == 'reflection':

        rs = (np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        rp = (n**2 * np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

        return rs,rp

    elif mode == 'transmission':

        ts = (2*np.cos(aoi))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        tp = (2*n*np.cos(aoi))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

        return ts,tp

    elif mode == 'both':

        ts = (2*np.cos(aoi))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        tp = (2*n*np.cos(aoi))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        rs = (np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))
        rp = (n**2 * np.cos(aoi) - np.sqrt(n**2 - np.sin(aoi)**2))/(n**2 * np.cos(aoi) + np.sqrt(n**2 - np.sin(aoi)**2))

        return rs,rp,ts,tp

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
def ConstructPRTMatrix(kin,kout,normal,aoi,n1,n2,mode='reflection'):
    normal = -normal

    # Compute the Fresnel coefficients for either transmission OR reflection
    fs,fp = FresnelCoefficients(aoi,n1,n2,mode=mode)

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

def GlobalToLocalCoordinates(Pmat,k,a=[0,1,0],exit_x=np.array([1,0,0])):

    # Double Pole Coordinate System, requires a rotation about an axis
    # Wikipedia article seems to disagree with CLY Example 11.4

    # Default entrance pupil for astronomical telescopes in Zemax
    O_e = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,-1]])

    # Compute Exit Pupil Basis Vectors
    # For arbitrary k each ray will have it's own pair of basis vectors
    # Get Exit Pupil Basis Vectors
    th = -np.arccos(np.dot(k,a))
    r = np.cross(k,a)

    ux = r[0]
    uy = r[1]
    uz = r[2]

    R11 = np.cos(th) + ux**2 *(1-np.cos(th))
    R12 = ux*uy*(1-np.cos(th)) - uz*np.sin(th)
    R13 = ux*uz*(1-np.cos(th)) + uy*np.sin(th)

    R21 = uy*ux*(1-np.cos(th)) + uz*np.sin(th)
    R22 = np.cos(th) + uy**2 *(1-np.cos(th))
    R23 = uy*uz*(1-np.cos(th)) - ux*np.sin(th)

    R31 = uz*ux*(1-np.cos(th)) - uy*np.sin(th)
    R32 = uz*uy*(1-np.cos(th)) + ux*np.sin(th)
    R33 = np.cos(th) + uz**2 * (1-np.cos(th))

    R = np.array([[R11,R12,R13],
                  [R21,R22,R23],
                  [R31,R32,R33]])

    # Local basis vectors
    x = R @ exit_x
    y = R @ np.cross(a,exit_x)
    # print('y')
    # print(y)
    O_x = np.array([x,y,k])

    # print('Rotation Matrix')
    # print(R)

    # print('O_entrance Pupil')
    # print(O_e)

    # print('O exit pupil')
    # print(O_x)

    # O_x = np.array([[1,0,0],
    #                 [0,0,1],
    #                 [0,-1,0]])
    O_x = R

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