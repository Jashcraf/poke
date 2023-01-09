# Tests the polarization functions
import poke.polarization as pol
import numpy as np
import pytest

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

    kin = np.array([0.,0.,1.],dtype='float64')
    kout = np.array([0.,1.,0.],dtype='float64')
    normal = np.sqrt(1/2)*np.array([0.,1.,1.],dtype='float64')

    Oinv,Oout = pol.ConstructOrthogonalTransferMatrices(kin,kout,normal)

    # Test Orthogonality
    np.testing.assert_allclose((Oinv.transpose(),Oout.transpose()),(np.linalg.inv(Oinv),np.linalg.inv(Oout)))

def test_ConstructPRTMatrix():
    """Ex 9.4 from Chipman, Lam, Young 2018
    """

    kin = np.array([0.,np.sin(np.pi/6),np.cos(np.pi/6)])
    eta = np.array([0.,0.,1.],dtype='float64')
    kout = np.array([0.,np.sin(np.pi/6),-np.cos(np.pi/6)])

    n1 = 1
    n2 = 1.5

    plate = {
        'surf':1, # this number doesn't really matter, just here for completeness
        'coating':n2,
        'mode':'reflect'
    }

    Ptest = np.array([[-0.240408,0,0],
                      [0,0.130825,0.501818],
                      [0,-0.501818,-0.710275]])

    P,J = pol.ConstructPRTMatrix(kin,kout,eta,np.arccos(np.dot(kin,eta)),plate,550e-9,n1)
    np.testing.assert_allclose(P,Ptest,rtol=1e-5)

@pytest.mark.skip(reason="example in text incorrect, looking for another")
def test_GlobalToLocalCoordinates():
    """Uses the double pole basis to rotate into local coordinates. 
    NOTE: In this investigation we actually discovered that the rotation matrix in this example is wrong!
    The basis vectors it returns are not orthogonal. But, we proceed with these vectors anyway.
    TODO: Maybe contact one of the authors to ask if they can update this example in future editions of the book?
    """

    pass

def test_JonesToMueller():

    """Example 6.11 in Chipman, Lam, Young
    """
    
    J = np.array([[1/4,1/4],
                  [1j/np.sqrt(2),-1j/np.sqrt(2)]])
    M = np.array([[9/16,0,-7/16,0],
                  [-7/16,0,9/16,0],
                  [0,0,0,-1/(2*np.sqrt(2))],
                  [0,-1/(2*np.sqrt(2)),0,0]])
                  
    Mtest = pol.JonesToMueller(J)
    np.testing.assert_allclose(M,Mtest,atol=1e-7)

@pytest.mark.skip(reason="low impact, no example in text")
def test_MuellerToJones():
    """No apparent example in Chipman
    TODO: write out an example using the same matrix in the previous test
    """
    pass

