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


pth = '/Users/jashcraft/Desktop/poke/tests/hst_rayfront_asbeamlets_32rays_1.65um.msgpack' # a 32 beamlet for the HST
rf = read_serial_to_rayfront(pth)

# set up detector coordinates
dsize = 0.5e-3
npix = 512
x = np.linspace(-dsize/2,dsize/2,npix)
x,y = np.meshgrid(x,x)
dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])
misalignbool = False

t1 = time.perf_counter()
field = rf.beamlet_decomposition_field(dcoords,misaligned=misalignbool).reshape([npix,npix])
t2 = time.perf_counter()

if misalignbool:
    method = 'new'
else:
    method = 'old'


plt.figure()
plt.suptitle(f'{method} method, t = {t2-t1}')
plt.subplot(121)
plt.imshow(np.log10(np.abs(field)**2))
plt.colorbar()
plt.subplot(122)
plt.imshow((np.angle(field)))
plt.colorbar()
plt.show()