import sys
sys.path.append('C:/Users/douglase/Desktop/poke/')
import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plt
from poke.gbd import *
import matplotlib.pyplot as plt
import time

# Initialize a Raybundle
nrays = 50
n1 = 1
n2 = 1##2.3669 - 1j*8.4177 # for subaru
pth = "C:/Users/douglase/Desktop/poke/test_files/Hubble_Test.zmx"

s1 = {
    'surf':1,
    'coating':1
}
s2 = {
    'surf':4,
    'coating':n2
}
s3 = {
    'surf':8,
    'coating':1
}

# Initialize 5 ray bundles - need to separate index calculation from raytracing
# p,m = plus, minus
# H,P = normalized field, normalized pupil

# Do some computation
wl = 1.65e-6
wo = 2.4*1.7/(2*nrays)
detsize = 1e-3
npix = 256
div = wl/(np.pi*wo)

dH = div/.08
dP = wo/1.2 

raybundle = pol.Rayfront(nrays,wl,1.2,0.08,circle=True)
raybundle.as_gaussianbeamlets(wo)
raybundle.TraceRaysetZOS(pth,surfaces=[s1,s3])
t1 = time.perf_counter()
field = raybundle.EvaluateGaussianField(detsize,npix)
t2 = time.perf_counter()
print(t2-t1,'s to compute gaussian field')

plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(np.log10(np.abs(field)**2))
plt.colorbar()
plt.subplot(122)
plt.imshow(np.angle(field),cmap='coolwarm')
plt.colorbar()
plt.show()


# raybundle_base = ray.Rayfront(nrays,n1,n2)
# raybundle_Hx = ray.Rayfront(nrays,n1,n2,dHx=dH)
# raybundle_Hy = ray.Rayfront(nrays,n1,n2,dHy=dH)
# raybundle_Px = ray.Rayfront(nrays,n1,n2,dPx=dP)
# raybundle_Py = ray.Rayfront(nrays,n1,n2,dPy=dP)


# raybool = False
# # try to run a raytrace
# raybundle_base.TraceThroughZOS(pth,surflist,global_coords=raybool)
# raybundle_Hx.TraceThroughZOS(pth,surflist,global_coords=raybool)
# raybundle_Hy.TraceThroughZOS(pth,surflist,global_coords=raybool)
# raybundle_Px.TraceThroughZOS(pth,surflist,global_coords=raybool)
# raybundle_Py.TraceThroughZOS(pth,surflist,global_coords=raybool)

# # Sweet now we have raydata, let's compute the differential ray transfer matrix
# dMat,O = ComputeDifferentialFromRaybundles(raybundle_base,raybundle_Px,raybundle_Py,raybundle_Hx,raybundle_Hy)

# # Generate the on-axis opd
# # raybundle_base.ComputeOPD()

# # Now evaluate the gaussian field
# rays = raybundle_base
# sys = dMat
# wavelength = 1.65e-6
# wo = 2.4*1.7/(2*51)
# detsize = 1778.080e-6 # stolen from zmx psf
# npix = 256
# Efield = eval_gausfield(rays,sys,wavelength,wo,detsize,npix,O)
# Efield = np.reshape(Efield,[npix,npix])

# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

# # Load zmx psf
# data = np.genfromtxt('Hubble_Test_FFTPSF_165um.txt',skip_header=18,encoding='UTF-16')

# plt.figure(figsize=[10,5])
# plt.subplot(121)
# plt.imshow(np.abs(data[int(npix - npix/2):int(npix + npix/2),int(npix - npix/2):int(npix + npix/2)]),norm=LogNorm())
# plt.title('ZMX FFT')
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(np.abs(Efield)/np.max(np.abs(Efield)),norm=LogNorm())
# plt.title('GBD PSF')
# plt.colorbar()
# plt.show()

# x = raybundle_base.xData[0]
# X = x
# y = raybundle_base.yData[0]
# Y = y 

# # x = x[X**2 + Y**2 <= 1.2]
# # y = y[X**2 + Y**2 <= 1.2]

# plt.figure()
# plt.scatter(x,y,c=raybundle_base.opd[-1])
# plt.title('OPD To Trace')
# plt.colorbar()
# plt.show()

# plt.figure(figsize=[10,10])

# plt.subplot(4,4,1)
# plt.scatter(x,y,c=dMat[0,0,:])
# plt.colorbar()
# plt.title('Axx')

# plt.subplot(4,4,2)
# plt.scatter(x,y,c=dMat[0,1,:])
# plt.colorbar()
# plt.title('Axy')

# plt.subplot(4,4,3)
# plt.scatter(x,y,c=dMat[0,2,:])
# plt.colorbar()
# plt.title('Bxx')

# plt.subplot(4,4,4)
# plt.scatter(x,y,c=dMat[0,3,:])
# plt.colorbar()
# plt.title('Bxy')

# plt.subplot(4,4,5)
# plt.scatter(x,y,c=dMat[1,0,:])
# plt.colorbar()
# plt.title('Ayx')

# plt.subplot(4,4,6)
# plt.scatter(x,y,c=dMat[1,1,:])
# plt.colorbar()
# plt.title('Ayy')

# plt.subplot(4,4,7)
# plt.scatter(x,y,c=dMat[1,2,:])
# plt.colorbar()
# plt.title('Byx')

# plt.subplot(4,4,8)
# plt.scatter(x,y,c=dMat[1,3,:])
# plt.colorbar()
# plt.title('Byy')

# plt.subplot(4,4,9)
# plt.scatter(x,y,c=dMat[2,0,:])
# plt.colorbar()
# plt.title('Cxx')

# plt.subplot(4,4,10)
# plt.scatter(x,y,c=dMat[2,1,:])
# plt.colorbar()
# plt.title('Cxy')

# plt.subplot(4,4,11)
# plt.scatter(x,y,c=dMat[2,2,:])
# plt.colorbar()
# plt.title('Dxx')

# plt.subplot(4,4,12)
# plt.scatter(x,y,c=dMat[2,3,:])
# plt.colorbar()
# plt.title('Dxy')

# plt.subplot(4,4,13)
# plt.scatter(x,y,c=dMat[3,0,:])
# plt.colorbar()
# plt.title('Cyx')

# plt.subplot(4,4,14)
# plt.scatter(x,y,c=dMat[3,1,:])
# plt.colorbar()
# plt.title('Cyy')

# plt.subplot(4,4,15)
# plt.scatter(x,y,c=dMat[3,2,:])
# plt.colorbar()
# plt.title('Dyx')

# plt.subplot(4,4,16)
# plt.scatter(x,y,c=dMat[3,3,:])
# plt.colorbar()
# plt.title('Dyy')

# plt.show()

