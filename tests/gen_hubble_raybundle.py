import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
import pickle

pth = 'C:/Users/douglase/Desktop/poke/test_files/Hubble_Test.zmx'
nrays = 30
wave = 1
global_coords = True

s1 = {
    'surf':2,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

s2 = {
    'surf':4,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

s3 = {
    'surf':8,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

surflist = [s1,s2,s3]

def GenHubbleRayfront():

    raybundle = Rayfront(nrays,1e-6,2.4,0.08,normalized_pupil_radius=1,fov=[0.,0.],circle=True)
    raybundle.as_polarized(surflist)
    raybundle.TraceRaysetZOS(pth)
    return raybundle
    
if __name__ == '__main__':
    
    rayfront = GenHubbleRayfront()
    with open('Hubble_Test_RayfrontZMX.pickle','wb') as f:
        pickle.dump(rayfront,f)