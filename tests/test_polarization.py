# Tests the polarization functions
import poke.polarization as pol
import numpy as np
import pytest

aoi = 10/180*np.pi
n1 = 1 # vacuum
n2 = 1.5075 # Schott BK7 at 1um 

def test_fresnel_coefficients():
    """Tests computation of the fresnel reflection and transmission coefficients
    """

    # hand computed values
    RS,RP = -0.2065274844798997, 0.19825093041916952
    TS,TP = 0.7934725155201003, 0.7948596553360993

    rs,rp = pol.FresnelCoefficients(aoi,n1,n2,mode='reflect')
    ts,tp = pol.FresnelCoefficients(aoi,n1,n2,mode='transmit')

    np.testing.assert_allclose((rs,rp,ts,tp),(RS,RP,TS,TP)) # default tolerance is 1e-7

def test_FresnelCoefficients_imag():
    pass

def test_orthogonal_transofrmation_matrices_relative():
    kin = np.array([0.,0.,1.],dtype=np.float64)
    kout = np.array([0.,1.,0.],dtype=np.float64)
    normal = np.sqrt(1/2)*np.array([0.,1.,1.],dtype=np.float64)

    Oinv,Oout = pol.orthogonal_transofrmation_matrices(kin,kout,normal)

    # Test Orthogonality
    np.testing.assert_allclose((Oinv.transpose(),Oout.transpose()),(np.linalg.inv(Oinv),np.linalg.inv(Oout)))

def test_orthogonal_transofrmation_matrices():
    """example 9.4 from chipman et al
    """

    kin = np.array([0.,np.sin(np.pi/6),np.cos(np.pi/6)])
    eta = np.array([0.,0.,1.])
    kout = np.array([0.,np.sin(np.pi/6),-np.cos(np.pi/6)])
    Oinv_ans = np.array([[1,0,0],
                         [0,np.sqrt(3)/2,-1/2],
                         [0,1/2,np.sqrt(3)/2]])
    
    Oout_ans = np.array([[1,0,0],
                         [0,-np.sqrt(3)/2,1/2],
                         [0,-1/2,-np.sqrt(3)/2]])

    Oinv,Oout = pol.orthogonal_transofrmation_matrices(kin,kout,eta)

    # Test Orthogonality
    np.testing.assert_allclose((Oinv,Oout),(Oinv_ans,Oout_ans))

def test_prt_matrix():
    """Ex 9.4 from Chipman, Lam, Young 2018
    """

    kin = np.array([0.,np.sin(np.pi/6),np.cos(np.pi/6)])
    eta = np.array([0.,0.,1.],dtype=np.float64)
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

    P,J,Q = pol.prt_matrix(kin,kout,eta,np.arccos(np.dot(kin,eta)),plate,550e-9,n1)
    np.testing.assert_allclose(P,Ptest,rtol=1e-5)

def test_vectorized_prt():

    # 10 prt matrices
    num_mat = 10
    kin = np.array([0.,np.sin(np.pi/6),np.cos(np.pi/6)]) * np.ones([num_mat,3])
    eta = np.array([0.,0.,1.],dtype=np.float64) * np.ones([num_mat,3])
    kout = np.array([0.,np.sin(np.pi/6),-np.cos(np.pi/6)]) * np.ones([num_mat,3])

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
    
    dprod = np.arccos(np.sum(kin*eta,axis=-1))
    Pvec,_,_ = pol.prt_matrix(kin,kout,eta,dprod,plate,550e-9,n1)
    Ptest = np.broadcast_to(Ptest,Pvec.shape)
    Ploop = np.zeros(Pvec.shape)
    for i in range(num_mat):

        Ploop[i],_,_ = pol.prt_matrix(kin[i],kout[i],eta[i],np.arccos(np.dot(kin[i],eta[i])),plate,550e-9,n1)
    
    np.testing.assert_allclose((Pvec,Ploop),(Ptest,Ptest),rtol=1e-5)

def test_global_to_local_coordinates():
    pass

def test_jones_to_mueller():
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
def test_mueller_to_jones():
    """No apparent example in Chipman
    TODO: write out an example using the same matrix in the previous test
    """
    pass

@pytest.mark.skip(reason="depreciated function")
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

@pytest.mark.skip(reason="depreciated function")
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