import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import matplotlib.pyplot as plt
import poke.plotting as plot
from astropy.io import fits

pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/test_files/PL&OS_CassegrainJonesPupil.zmx'
nrays = 32
wave = 1
global_coords = True
nloops = 5
d_film = 1e-8

n_Al = 1.2 + 1j*7.115 # 600nm from CV Al coating MUL
n_Ag = 0.055159 + 1j*4.0097
n_SiO2 = 1.4580 + 2*1j
n_BK7 = 1
wvl = 600e-9
i=0

s1 = {
    'surf':1,
    'coating':[(n_SiO2,d_film*(i+1)),(n_BK7,d_film*(i+0.5)),n_Al],
    'mode':'reflect'
}

s2 = {
    'surf':2,
    'coating':[(n_SiO2,d_film*(i+1)),(n_BK7,d_film*(i+0.5)),n_Al],
    'mode':'reflect'
}

surflist = [s1,s2]

rf = Rayfront(nrays,wvl,1.2,.08)
rf.as_polarized(surflist)
rf.trace_rayset(pth)

for i in range(1,nloops):

    rf.compute_jones_pupil(aloc=np.array([0.,0.,1.]))
    plot.jones_pupil(rf)

    s1 = {
        'surf':1,
        'coating':[(n_SiO2,d_film*(i+1)),(n_BK7,d_film*(i+0.5)),n_Al],
        'mode':'reflect'
    }

    s2 = {
        'surf':2,
        'coating':[(n_SiO2,d_film*(i+1)),(n_BK7,d_film*(i+0.5)),n_Al],
        'mode':'reflect'
    }

    # update surfacelist
    rf.surfaces = [s1,s2]

print(len(rf.jones_pupil))

# does the xx element change at all?
x,y = rf.xData[0,0],rf.yData[0,0]
plt.figure()
plt.scatter(x,y,c=np.angle(rf.jones_pupil[0][...,0,0])-np.angle(rf.jones_pupil[3][...,0,0]))
plt.colorbar()
plt.show()