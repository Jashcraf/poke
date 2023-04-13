import numpy as np
import sys
import pickle
sys.path.append('/Users/jashcraft/Desktop/poke')
pth = '/Users/jashcraft/Desktop/poke/test_files/Hubble_Test_RayfrontZMX.pickle'

with open(pth,'rb') as f:
    rayfront = pickle.load(f)
