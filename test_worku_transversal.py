import poke.gbd as gbd
#import poke.worku_gbd as worku
import numpy as np
import poke.raytrace as ray
import matplotlib.pyplot as plt

# Initialize a Raybundle
nrays = 50
npix = 128
detsize = 1e-3
n1 = 1
n2 = 1 # 2.3669 + 1j*8.4177 # for subaru
pth = "C:/Users/UASAL-OPTICS/Desktop/poke/Hubble_Test.zmx"
surflist = [2,4,8]

# Initialize 5 ray bundles - need to separate index calculation from raytracing
# p,m = plus, minus
# H,P = normalized field, normalized pupil
wl = 1.65e-6
wo = .04
dh = wl/(np.pi*wo)
dH = dh/.08 * 180 / np.pi
dP = wo/2.4 

raybundle_base = ray.Rayfront(nrays,n1,n2,wl)
raybundle_Px = ray.Rayfront(nrays,n1,n2,wl,dPx=dP)
raybundle_Py = ray.Rayfront(nrays,n1,n2,wl,dPy=dP)
raybundle_Hx = ray.Rayfront(nrays,n1,n2,wl,dHx=dH)
raybundle_Hy = ray.Rayfront(nrays,n1,n2,wl,dHy=dH)


raybool = False
# try to run a raytrace
raybundle_base.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Px.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Py.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Hx.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Hy.TraceThroughZOS(pth,surflist,global_coords=raybool)

# this one sorta works
# Field = gbd.eval_gausfield_worku(raybundle_base,raybundle_Px,raybundle_Py,raybundle_Hx,raybundle_Hy,
#                          wl,wo,detsize,npix,
#                          wo,wo,dh,dh,
#                          detector_normal=np.array([0,0,1]))

Field = gbd.EvalGausfieldWorku(raybundle_base,raybundle_Px,raybundle_Py,raybundle_Hx,raybundle_Hy,
                         wl,wo,detsize,npix,
                         wo,wo,np.tan(dh),np.tan(dh),
                         detector_normal=np.array([0,0,1]))
                         
from matplotlib.colors import LogNorm

pth_to_zmx_psf = 'C:/Users/UASAL-OPTICS/Desktop/poke/Hubble_Test_FFTPSF_165um_tilted.txt'
pth_to_zmx_psf = 'C:/Users/UASAL-OPTICS/Desktop/poke/Hubble_Test_FFTPSF_165um_defocused.txt'
zmxpsf = np.genfromtxt(pth_to_zmx_psf,encoding='UTF-16',skip_header=18)
os = 2

from scipy.ndimage import zoom
# zmxpsf = zmxpsf[int(zmx_center-npix/2):int(zmx_center+npix/2),int(zmx_center-npix/2):int(zmx_center+npix/2)]
zmxpsf = zoom(zmxpsf,1/os)
zmx_center = int(zmxpsf.shape[0]/2)
zmxpsf = zmxpsf[int(zmx_center-npix/2):int(zmx_center+npix/2),int(zmx_center-npix/2):int(zmx_center+npix/2)]


norm_field = np.abs(Field)
norm_field /= np.max(norm_field)

zmxpsf /= np.max(zmxpsf)
zmxpsf = np.rot90(np.rot90(zmxpsf))

plt.figure(figsize=[15,5])
plt.subplot(131)
plt.title('GBD PSF')
plt.imshow(norm_field,norm=LogNorm(),cmap='magma',origin='lower')
plt.colorbar()

plt.subplot(132)
plt.title('ZMX PSF')
plt.imshow(zmxpsf,norm=LogNorm(),cmap='magma',origin='lower')
plt.colorbar()

plt.subplot(133)
plt.title('Fractional Difference')
plt.imshow((norm_field-zmxpsf)/zmxpsf,cmap='RdBu')
plt.colorbar()
plt.show()
                         
