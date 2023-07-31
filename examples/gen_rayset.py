from poke.poke_core import Rayfront
import poke.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
from poke.writing import write_rayfront_to_serial

# Initialize a Raybundle
nrays = 64
n1 = 1
n2 = 1.0194 + 1j*6.6388 # Al in v band
radius = 1.2
ffov = 0.08
wlen = 551e-9
pth = "C:/Users/UASAL-OPTICS/Desktop/poke/test_files/Hubble_Test.zmx"

# Surface parameters are defined with Python dictionaries.
# 'surf' is the surface number in the raytrace lens data editor
# 'coating' is the thin film information. For a single layer, it's just the refractive index
s1 = {
    'surf':2,
    'coating':n2,
    'mode':'reflect'
}

s2 = {
    'surf':4,
    'coating':n2,
    'mode':'reflect'
}

s3 = {
    'surf':8,
    'coating':n2,
    'mode':'reflect'
}

# Initialize a Rayfront
rf = Rayfront(nrays,wlen,radius,ffov,circle=False)
rf.as_polarized([s1,s2]) # pass the raybundle the surface list

# Trace the rays through a zemax optical system
rf.trace_rayset(pth)

write_rayfront_to_serial(rf,'sample_rayfront')

