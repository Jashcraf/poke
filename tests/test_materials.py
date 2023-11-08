import numpy as np
import poke.materials as mat
import pytest
import os

def test_create_index_model():

    # TODO: The file import system doesn't work on the GH Actions test 

    # prealloc lists
    reference = []
    test = []

    # pick a complex material and a dielectric
    materials = ['Al','HfO2']

    # load refractive indices from a certain wavelength
    wvl = [0.51660,0.5060]
    ns = [0.51427,1.9083754098692]
    ks = [4.95111,0]
    for material,wave,n,k in zip(materials,wvl,ns,ks):
        n_callable = mat.create_index_model(material)

        reference.append(n + 1j*k)
        test.append(n_callable(wave))
    
    # np.testing.assert_allclose(reference,test)
    pass
