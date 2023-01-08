# run tests
# from poke.pythonraytrace import *
import poke.poke_core as pol
import numpy as np

# Hubble Path
system_pth = "'C:/Users/jaren/Desktop/Polarization-Raytrace/raytrace-files/"
fn = "Hubble_Text.zmx"

pth = system_pth + fn

# Test Orthogonal Transformation
def testOmat():
    # Ex 10.3 CLY
    # Two gold-coated fold , this is for the first 
    # tests that the orthogonal transfer matrices are being done correctly, not the ray tracing or material calculation

    n1 = np.array([0,1,1])/np.sqrt(2)
    n2 = np.array([-1,-1,0])/np.sqrt(2)

    k1 = np.array([-0.195,-0.195,0.961])
    k2 = np.array([-0.195,-0.961,0.195])

    Oinv,Oout = pol.ConstructOrthogonalTransferMatrices(k1,k2,n1)
    # Oinv = np.round(Oinv,4)
    # Oout = np.round(Oout,4)

    s1 = np.array([-0.973,0.164,-0.164])
    p1 = np.array([-0.126,-0.967,-0.221])

    s2 = np.array([-0.973,0.164,-0.164])
    p2 = np.array([0.126,-0.221,-0.967])

    oinv = np.array([s1,p1,k1])
    oout = np.transpose(np.array([s2,p2,k2]))

    print('Oinv reference = ')
    print(oinv)
    print('Oinv test = ')
    print(Oinv)
    print('Oinv difference = ')
    print(oinv-Oinv)

    print('Oout reference = ')
    print(oout)
    print('Oout test = ')
    print(Oout)
    print('Oout difference = ')
    print(oout-Oout)

    # assert (Oinv == oinv).all(), "Inverse calculation is wrong"
    # assert (Oout == oout).all(), "Outgoing calculation incorrect"

def testFcoeffs():
    
    # Aluminum at 750nm at 45 degrees
    n1 = 1
    n2 = 2.3669 + 1j*8.4177

    aoi = np.pi/4

    rs,rp = pol.FresnelCoefficients(aoi,n1,n2)

    Rs = np.abs(rs)**2
    Rp = np.abs(rp)**2

    Rp_cal = 0.84151
    Rs_cal = 0.91734

    print('Rs reference = ')
    print(Rs_cal)
    print('Rs test = ')
    print(Rs)
    print('Rs difference = ')
    print(Rs_cal-Rs)
    
    print('Rp reference = ')
    print(Rp_cal)
    print('Rp test = ')
    print(Rp)
    print('Rp difference = ')
    print(Rp_cal-Rp)

def testPRTmatrix():

    # Aluminum at 750nm at 45 degrees
    n1 = 1
    n2 = 2.3669 + 1j*8.4177
    aoi = np.pi/4

    norm = np.array([0,1,1])/np.sqrt(2)

    k1 = np.array([-0.195,-0.195,0.961])
    k2 = np.array([-0.195,-0.961,0.195])

    Ptest,J = pol.ConstructPRTMatrix(k1,k2,norm,aoi,n1,n2)

    Pcal = np.array([[-0.889 + 1j*0.219,0.106 + 1j*0.046, -0.361 + 1j*0.054],
                     [0.361-1j*0.054,0.314-1j*0.137,-0.863-1j*0.039],
                     [-0.106-1j*0.046,0.655-1j*0.628,0.314-1j*0.137]])

    print('P reference')
    print(Pcal)
    print('P test')
    print(Ptest)
    print('P Difference')
    print(Pcal-Ptest)

testOmat()
testFcoeffs()
testPRTmatrix()


