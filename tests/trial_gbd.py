import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

sys.path.append('C:/Users/UASAL-OPTICS/Desktop/poke')
from poke.poke_core import Rayfront



pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/tests/Hubble_Test.zmx'
nrays = 26
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
    'surf':8,
    'mode':'reflect',
    'coating':0.04 + 1j*7
}
surflist = [s1,si]

# set up detector coordinates
dsize = 1e-3
npix = 64
x = np.linspace(-dsize/2,dsize/2,npix)
x,y = np.meshgrid(x,x)
dcoords = np.array([x.ravel(),y.ravel(),0*x.ravel()])

rf = Rayfront(nrays,wavelength,pupil_radius,max_fov,waist_pad=wo)
rf.as_gaussianbeamlets(wo)
rf.TraceRaysetZOS(pth,surfaces=surflist)

field = rf.EvaluateGaussianField(dsize,npix)

plt.figure()
plt.imshow(np.log10(np.abs(field)**2))
plt.colorbar()
plt.show()
