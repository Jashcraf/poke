import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import pickle

sys.path.append('C:/Users/ashcraft/Desktop/poke')
from poke.poke_core import Rayfront



pth = 'C:/Users/ashcraft/Desktop/poke/tests/hubble_test.len'
nrays = 10
wavelength = 1.65e-6
pupil_radius = 1.2
max_fov = 0.08
OF = 1.4
wo = 2*pupil_radius*OF / (2*nrays)

# set up surfaces
s1 = {
    'surf':1,
    'mode':'reflect',
    'coating':0.04 + 1j*7
}
si = {
    'surf':6,
    'mode':'reflect',
    'coating':0.04 + 1j*7
}
surflist = [s1,si]

# set up detector coordinates
dsize = 1e-3
npix = 32
x = np.linspace(-dsize/2,dsize/2,32)
x,y = np.meshgrid(x,x)
dcoords = np.array([x.ravel(),y.ravel(),0*x.ravel()])

rf = Rayfront(nrays,wavelength,pupil_radius,max_fov)
rf.as_gaussianbeamlets(wo)
rf.trace_rayset(pth,surfaces=surflist)
# with open (f'test_hst_rayfront_gauslets_{nrays}beams_1.65um.pickle','wb') as f:
#     pickle.dump(rf,f)

plt.figure()
plt.scatter(rf.xData[:,-1],rf.yData[:,-1],c=rf.opd[:,-1])
plt.colorbar()
plt.show()

# field = rf.beamlet_decomposition_field(dcoords).reshape([npix,npix])

# plt.figure()
# plt.imshow(np.abs(field))
# plt.colorbar()
# plt.show()