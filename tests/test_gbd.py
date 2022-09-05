import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plt
from poke.gbd import *

# Initialize a Raybundle
nrays = 50
n1 = 1
n2 = 2.3669 + 1j*8.4177 # for subaru
pth = "C:/Users/LOFT_Olaf/Desktop/poke/Hubble_Test.zmx"
surflist = [2,4,7]

# Initialize 5 ray bundles - need to separate index calculation from raytracing
# p,m = plus, minus
# H,P = normalized field, normalized pupil

dH = 1.65e-6/(np.pi*.04)/.08
dP = .04/2.4 

raybundle_base = ray.Rayfront(nrays,n1,n2)
raybundle_Hx = ray.Rayfront(nrays,n1,n2,dHx=dH)
raybundle_Hy = ray.Rayfront(nrays,n1,n2,dHy=dH)
raybundle_Px = ray.Rayfront(nrays,n1,n2,dPx=dP)
raybundle_Py = ray.Rayfront(nrays,n1,n2,dPy=dP)


raybool = False
# try to run a raytrace
raybundle_base.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Hx.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Hy.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Px.TraceThroughZOS(pth,surflist,global_coords=raybool)
raybundle_Py.TraceThroughZOS(pth,surflist,global_coords=raybool)

# Sweet now we have raydata, let's compute the differential ray transfer matrix
dMat,O = ComputeDifferentialFromRaybundles(raybundle_base,raybundle_Px,raybundle_Py,raybundle_Hx,raybundle_Hy)

# Generate the on-axis opd
# raybundle_base.ComputeOPD()

# Now evaluate the gaussian field
rays = raybundle_base
sys = dMat
wavelength = 1.65e-6
wo = 2.4*1.7/(2*51)
detsize = 1778.080e-6 # stolen from zmx psf
npix = 256
Efield = eval_gausfield(rays,sys,wavelength,wo,detsize,npix,O)
Efield = np.reshape(Efield,[npix,npix])

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load zmx psf
data = np.genfromtxt('Hubble_Test_FFTPSF_165um.txt',skip_header=18,encoding='UTF-16')

plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(np.abs(data[int(npix - npix/2):int(npix + npix/2),int(npix - npix/2):int(npix + npix/2)]),norm=LogNorm())
plt.title('ZMX FFT')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.abs(Efield)/np.max(np.abs(Efield)),norm=LogNorm())
plt.title('GBD PSF')
plt.colorbar()
plt.show()

x = raybundle_base.xData[0]
X = x
y = raybundle_base.yData[0]
Y = y 

# x = x[X**2 + Y**2 <= 1.2]
# y = y[X**2 + Y**2 <= 1.2]

plt.figure()
plt.scatter(x,y,c=raybundle_base.opd[-1])
plt.title('OPD To Trace')
plt.colorbar()
plt.show()

plt.figure(figsize=[10,10])

plt.subplot(4,4,1)
plt.scatter(x,y,c=dMat[0,0,:])
plt.colorbar()
plt.title('Axx')

plt.subplot(4,4,2)
plt.scatter(x,y,c=dMat[0,1,:])
plt.colorbar()
plt.title('Axy')

plt.subplot(4,4,3)
plt.scatter(x,y,c=dMat[0,2,:])
plt.colorbar()
plt.title('Bxx')

plt.subplot(4,4,4)
plt.scatter(x,y,c=dMat[0,3,:])
plt.colorbar()
plt.title('Bxy')

plt.subplot(4,4,5)
plt.scatter(x,y,c=dMat[1,0,:])
plt.colorbar()
plt.title('Ayx')

plt.subplot(4,4,6)
plt.scatter(x,y,c=dMat[1,1,:])
plt.colorbar()
plt.title('Ayy')

plt.subplot(4,4,7)
plt.scatter(x,y,c=dMat[1,2,:])
plt.colorbar()
plt.title('Byx')

plt.subplot(4,4,8)
plt.scatter(x,y,c=dMat[1,3,:])
plt.colorbar()
plt.title('Byy')

plt.subplot(4,4,9)
plt.scatter(x,y,c=dMat[2,0,:])
plt.colorbar()
plt.title('Cxx')

plt.subplot(4,4,10)
plt.scatter(x,y,c=dMat[2,1,:])
plt.colorbar()
plt.title('Cxy')

plt.subplot(4,4,11)
plt.scatter(x,y,c=dMat[2,2,:])
plt.colorbar()
plt.title('Dxx')

plt.subplot(4,4,12)
plt.scatter(x,y,c=dMat[2,3,:])
plt.colorbar()
plt.title('Dxy')

plt.subplot(4,4,13)
plt.scatter(x,y,c=dMat[3,0,:])
plt.colorbar()
plt.title('Cyx')

plt.subplot(4,4,14)
plt.scatter(x,y,c=dMat[3,1,:])
plt.colorbar()
plt.title('Cyy')

plt.subplot(4,4,15)
plt.scatter(x,y,c=dMat[3,2,:])
plt.colorbar()
plt.title('Dyx')

plt.subplot(4,4,16)
plt.scatter(x,y,c=dMat[3,3,:])
plt.colorbar()
plt.title('Dyy')

plt.show()

