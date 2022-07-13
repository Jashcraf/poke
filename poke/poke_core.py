# poke core functions
# This is the meat of the PRT calculation

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
    # kin /= np.linalg.norm(kin) # these were not in chippman and lam - added 03/30/2022
    # kout /= np.linalg.norm(kout)

    sin = np.cross(kin,normal)
    sin /= np.linalg.norm(sin) # normalize the s-vector
    pin = np.cross(kin,sin)
    Oinv = np.array([sin,pin,kin])

    sout = sin #np.cross(kout,normal)
    sout /= np.linalg.norm(sout) # normalize the s-vector
    pout = np.cross(kout,sout)
    Oout = np.transpose(np.array([sout,pout,kout]))

    return Oinv,Oout

# Step 3) Create Polarization Ray Trace matrix
def ConstructPRTMatrix(kin,kout,normal,aoi,n1,n2,mode='reflection'):

    # Compute the Fresnel coefficients for either transmission OR reflection
    fs,fp = FresnelCoefficients(aoi,n1,n2,mode=mode)

    # Compute the orthogonal transfer matrices
    Oinv,Oout = ConstructOrthogonalTransferMatrices(kin,kout,normal)

    # Compute the Jones matrix
    J = np.array([[fs,0,0],[0,fp,0],[0,0,1]])

    # Compute the Polarization Ray Tracing Matrix
    # Pmat = np.matmul(Oout,np.matmul(J,Oinv))
    Pmat = Oout @ J @ Oinv

    # This returns the polarization ray tracing matrix but I'm not 100% sure its in the coordinate system of the Jones Pupil

    return Pmat,J