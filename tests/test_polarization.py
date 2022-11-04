# Tests the polarization functions

import poke.polarization as pol
import numpy as np

aoi = 10/180*np.pi
n1 = 1 # vacuum
n2 = 1.5075 # Schott BK7 at 1um 

def test_FresnelCoefficients():
    """Tests computation of the fresnel reflection and transmission coefficients
    """

    # hand computed values
    RS,RP = -0.2065274844798997, 0.19825093041916952
    TS,TP = 0.7934725155201003, 0.7948596553360993

    rs,rp = pol.FresnelCoefficients(aoi,n1,n2,mode='reflect')
    ts,tp = pol.FresnelCoefficients(aoi,n1,n2,mode='transmit')

    np.testing.assert_allclose((rs,rp,ts,tp),(RS,RP,TS,TP)) # default tolerance is 1e-7


def test_ConstructOrthogonalTransferMatrices():
    """utilizes properties of orthogonal matrices to test orthogonality,
    inspired by a test written by Quinn Jarecki
    """

    kin = np.array([0,0,1])
    kout = np.array([0,1,0])
    normal = np.sqrt(1/2)*np.array([0,1,1])

    Oinv,Oout = pol.ConstructOrthogonalTransferMatrices(kin,kout,normal)

    # Test Orthogonality
    np.testing.assert_allclose((Oinv.transpose(),Oout.transpose()),(np.linalg.inv(Oinv),np.linalg.inv(Oout)))

def test_ConstructPRTMatrix():
    pass

def test_GlobalToLocalCoordinates():
    pass

def test_JonesToMueller():
    pass

def MuellerToJones():
    pass

