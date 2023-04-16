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

def savefits(array,fn):

    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fn,overwrite=True)

pth_to_brr = "C:/Users/UASAL-OPTICS/Desktop/gbd_go_brr/"

t0 = time.perf_counter()
pth = '/Users/UASAL-OPTICS/Desktop/poke/test_files/Hubble_Test.zmx' # a 32 beamlet for the HST
nrays = [32]# [400,500,600]
wavelength = 1.65e-6
pupil_radius = 1.2
max_fov = .08
OF = 2

for rays in nrays:
        
    rf = Rayfront(rays,wavelength,pupil_radius,max_fov,grid='fib')
    wo = 2*pupil_radius*OF / (2*rays)
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
    dsize = 1e-3#0.007920000012478126
    npix = 64#1600
    x = np.linspace(-dsize/2,dsize/2,npix)
    x,y = np.meshgrid(x,x)
    dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])
    misalignbool = True

    t1 = time.perf_counter()
    field = rf.beamlet_decomposition_field(dcoords,misaligned=misalignbool,memory_avail=4).reshape([npix,npix])
    t2 = time.perf_counter()

    if misalignbool:
        method = 'new'
    else:
        method = 'old'

    plt.figure()
    plt.imshow(np.log10(np.abs(field)))
    plt.colorbar()
    plt.show()

    #savefits(np.abs(field),pth_to_brr+f'hst_onax_even_{rays}beams_OF{OF}_{int(dsize*1e3)}mm_{npix}pix_abs.fits')
    #savefits(np.angle(field),pth_to_brr+f'hst_onax_even_{rays}beams_OF{OF}_{int(dsize*1e3)}mm_{npix}pix_angle.fits')

print(f'time to perform all simulations = {time.perf_counter()-t0}')