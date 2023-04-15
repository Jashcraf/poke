import sys
# sys.path.append('/Users/jashcraft/Desktop/poke')
from poke.poke_math import np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import pickle
from poke.poke_core import Rayfront
from poke.writing import read_serial_to_rayfront
import time


pth = '/Users/UASAL-OPTICS/Desktop/poke/test_files/Hubble_Test.zmx' # a 32 beamlet for the HST
nrays = 150
wavelength = 1.65e-6
pupil_radius = 1.2
max_fov = .08
rf = Rayfront(nrays,wavelength,pupil_radius,max_fov)
OF = 1.4
wo = 2*pupil_radius*OF / (2*nrays)
rf.as_gaussianbeamlets(wo)

s1 = {
    'surf':1
}
si = {
    'surf':8
}
surflist = [s1,si]
rf.trace_rayset(pth,surfaces=surflist)

# set up detector coordinates
dsize = 0.5e-3
npix = 128
x = np.linspace(-dsize/2,dsize/2,npix)
x,y = np.meshgrid(x,x)
dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])
misalignbool = True

t1 = time.perf_counter()
field = rf.beamlet_decomposition_field(dcoords,misaligned=misalignbool,memory_avail=8).reshape([npix,npix])
t2 = time.perf_counter()

if misalignbool:
    method = 'new'
else:
    method = 'old'


plt.figure()
plt.suptitle(f'{method} method, t = {t2-t1}')
plt.subplot(121)
plt.imshow(np.log10(np.abs(field)**2),cmap='gray')
plt.colorbar()
plt.subplot(122)
plt.imshow((np.angle(field)))
plt.colorbar()
plt.show()