import poke.gbd as gbd
#import poke.worku_gbd as worku
import numpy as np
import poke.raytrace as ray
import matplotlib.pyplot as plt
from astropy.io import fits

# Initialize a Raybundle
nrays = 150
npix = 256
detsize = 1e-3
n1 = 1
n2 = 1 # 2.3669 + 1j*8.4177 # for subaru
pth = "C:/Users/LOFT_Olaf/Desktop/poke/Hubble_Test.zmx"
surflist = [1,8]

# Initialize 5 ray bundles - need to separate index calculation from raytracing
# p,m = plus, minus
# H,P = normalized field, normalized pupil
wl = 1.65e-6
OF = 1.7
wo = OF*2.4/(2*nrays)
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

hdu = fits.PrimaryHDU(np.abs(Field))
hdul = fits.HDUList(hdu)
hdul.writeto('GBD_Paper/gbdpsf_165um_unobscured_OF17_150beams_1mmdet.fits')

