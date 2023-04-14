import sys
# sys.path.append('/Users/jashcraft/Desktop/poke')
from poke.poke_math import np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import pickle
from poke.poke_core import Rayfront
from poke.writing import read_serial_to_rayfront


pth = '/Users/jashcraft/Desktop/poke/tests/hst_rayfront_asbeamlets_32rays_1.65um.msgpack' # a 32 beamlet for the HST
rf = read_serial_to_rayfront(pth)

# set up detector coordinates
dsize = 0.5e-3
npix = 128
x = np.linspace(-dsize/2,dsize/2,npix)
x,y = np.meshgrid(x,x)
dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])

field = rf.beamlet_decomposition_field(dcoords,misaligned=False).reshape([npix,npix])

plt.figure()
plt.subplot(121)
plt.imshow(np.log10(np.abs(field)**2))
plt.colorbar()
plt.subplot(122)
plt.imshow((np.angle(field)))
plt.colorbar()
plt.show()