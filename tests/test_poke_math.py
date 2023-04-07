import numpy as np
import poke.poke_math as pmath
import pytest

# np.testing.assert_allclose((rs,rp,ts,tp),(RS,RP,TS,TP)) # default tolerance is 1e-7


def test_det_2x2():

    array = np.array([[1,2],[3,4]])
    answer = 1*4-2*3
    test = pmath.det_2x2(array)

    np.testing.assert_allclose(answer,test)

def test_det_2x2_broadcasted():

    # going to check that the looped version is the same as the broadcasted
    nloops = 10
    array = np.array([[1,2],[3,4]])
    answer = 1*4-2*3
    answer_broadcasted = np.ones(nloops)*answer
    array_broadcasted = np.broadcast_to(array,[nloops,*array.shape])
    array_looped = np.zeros_like(answer_broadcasted)
    determinant_test = pmath.det_2x2(array_broadcasted)
    
    for i in range(nloops):
        array_looped[i] = pmath.det_2x2(array)

    np.testing.assert_allclose((determinant_test,array_looped),(answer_broadcasted,answer_broadcasted))

def test_mat_inv_2x2():

    array = np.array([[1,2],[3,4]])
    answer = np.array([[-2,1],[1.5,-0.5]])
    test = pmath.mat_inv_2x2(array)

    np.testing.assert_allclose(answer,test)

def test_mat_inv_2x2_broadcasted():

    # going to check that the looped version is the same as the broadcasted
    nloops = 10
    array = np.array([[1,2],[3,4]])
    answer = np.array([[-2,1],[1.5,-0.5]])
    answer_broadcasted = np.broadcast_to(answer,[nloops,*array.shape])
    array_broadcasted = np.broadcast_to(array,[nloops,*array.shape])
    array_looped = np.zeros_like(answer_broadcasted)
    inv_test = pmath.mat_inv_2x2(array_broadcasted)
    
    for i in range(nloops):
        array_looped[i] = pmath.mat_inv_2x2(array)

    np.testing.assert_allclose((inv_test,array_looped),(answer_broadcasted,answer_broadcasted))

def test_mat_inv_3x3():

    array = np.array([[1,-1,1],[-1,-1,1],[-1,1,1]])
    answer= np.array([[0.5,-0.5,0],[0,-0.5,0.5],[0.5,0,0.5]])
    test = pmath.mat_inv_3x3(array)

    np.testing.assert_allclose(test,answer)

def test_mat_inv_3x3_broadcasted():

    # going to check that the looped version is the same as the broadcasted
    nloops = 10
    array = np.array([[1,-1,1],[-1,-1,1],[-1,1,1]])
    answer= np.array([[0.5,-0.5,0],[0,-0.5,0.5],[0.5,0,0.5]])
    answer_broadcasted = np.broadcast_to(answer,[nloops,*array.shape])
    array_broadcasted = np.broadcast_to(array,[nloops,*array.shape])
    array_looped = np.zeros(array_broadcasted.shape)
    inv_test = pmath.mat_inv_3x3(array_broadcasted)
    
    for i in range(nloops):
        print(f'{i}th loop done')
        array_looped[i] = pmath.mat_inv_3x3(array)
        print(pmath.mat_inv_3x3(array))
        print(array_looped[i])

    np.testing.assert_allclose((inv_test,array_looped),(answer_broadcasted,answer_broadcasted))

def test_eigenvalues_2x2():

    # eigenvalues of a circular polarizer
    circ = np.array([[1,1j],[-1j,1]])/2

    answer1,answer2 = 1,0
    test1,test2 = pmath.eigenvalues_2x2(circ)

    np.testing.assert_allclose((test1,test2),(answer1,answer2))

def test_eigenvalues_2x2_broadcasted():

    # going to check that the looped version is the same as the broadcasted
    nloops = 10
    circ = np.array([[1,1j],[-1j,1]])/2
    circ_broadcasted = np.broadcast_to(circ,[nloops,*circ.shape])
    answer1 = np.broadcast_to(1 + 0*1j,nloops)
    answer2 = np.broadcast_to(0*1j,nloops)

    answer_looped1 = np.zeros_like(answer1)
    answer_looped2 = np.zeros_like(answer2)
    test1,test2 = pmath.eigenvalues_2x2(circ_broadcasted)
    
    for i in range(nloops):
        answer_looped1[i],answer_looped2[i] = pmath.eigenvalues_2x2(circ)

    np.testing.assert_allclose((test1,test2,answer_looped1,answer_looped2),(answer1,answer2,answer1,answer2))

def test_vector_norm():

    vector = np.array([1,1,1])
    answer = np.sqrt(3)
    test = pmath.vector_norm(vector)

    np.testing.assert_allclose(test,answer)

def test_vector_norm_broadcasted():

    nloops = 10
    vector = np.array([1,1,1])
    vector_broadcasted = np.broadcast_to(vector,[nloops,*vector.shape])
    answer = np.broadcast_to(np.sqrt(3),nloops)
    vector_looped = np.zeros(answer.shape)

    for i in range(nloops):
        vector_looped[i] = pmath.vector_norm(vector)

    test = pmath.vector_norm(vector_broadcasted)

    np.testing.assert_allclose((vector_looped,test),(answer,answer))

def test_vector_angle():

    v1 = np.array([0,0,1])
    v2 = np.array([0,1,1])/np.sqrt(2)
    answer = np.pi/4
    test = pmath.vector_angle(v1,v2)

    np.testing.assert_allclose(test,answer)

def test_vector_angle_broadcasted():

    nloops = 10
    v1 = np.array([0,0,1])
    v2 = np.array([0,1,1])/np.sqrt(2)
    v2_broadcast = np.broadcast_to(v2,[nloops,*v2.shape])
    answer = np.pi/4
    answer_broadcast = np.broadcast_to(answer,nloops)
    test_looped = np.zeros(answer_broadcast.shape)
    test = pmath.vector_angle(v1,v2_broadcast)
    for i in range(nloops):
        test_looped[i] = pmath.vector_angle(v1,v2)

    np.testing.assert_allclose((test,test_looped),(answer_broadcast,answer_broadcast))


def test_rotation_3d():

    angle = np.pi/4
    axis = np.array([0.,0.,1.])
    s2 = 1/np.sqrt(2)
    answer = np.array([[s2,-s2,0],[s2,s2,0],[0,0,1]])
    test = pmath.rotation_3d(angle,axis)

    np.testing.assert_allclose(answer,test)

def test_rotation_3d_broadcasted():
    
    nloops = 10
    angle = np.pi/4
    axis = np.array([0.,0.,1.])
    s2 = 1/np.sqrt(2)
    answer = np.array([[s2,-s2,0],[s2,s2,0],[0,0,1]])

    answer_broadcasted = np.broadcast_to(answer,[nloops,*answer.shape])
    axis_broadcasted = np.broadcast_to(axis,[nloops,*axis.shape])
    angle_broadcasted = np.broadcast_to(angle,nloops)

    test_broadcasted = pmath.rotation_3d(angle_broadcasted,axis_broadcasted)

    np.testing.assert_allclose(test_broadcasted,answer_broadcasted)



