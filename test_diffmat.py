import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plt
from poke.gbd import *

# Initialize a Raybundle
nrays = 32
n1 = 1
n2 = 2.3669 + 1j*8.4177 # for subaru
pth = "C:/Users/jaren/Desktop/poke/Hubble_Test.zmx"
surflist = [2,4,7]

# Initialize 5 ray bundles - need to separate index calculation from raytracing
# p,m = plus, minus
# H,P = normalized field, normalized pupil

dH = 1e-6
dP = 1e-6

raybundle_base = ray.RayBundle(nrays,n1,n2)
raybundle_Hx = ray.RayBundle(nrays,n1,n2,dHx=dH)
raybundle_Hy = ray.RayBundle(nrays,n1,n2,dHy=dH)
raybundle_Px = ray.RayBundle(nrays,n1,n2,dPx=dP)
raybundle_Py = ray.RayBundle(nrays,n1,n2,dPy=dP)


# try to run a raytrace
raybundle_base.TraceThroughZOS(pth,surflist)
raybundle_Hx.TraceThroughZOS(pth,surflist)
raybundle_Hy.TraceThroughZOS(pth,surflist)
raybundle_Px.TraceThroughZOS(pth,surflist)
raybundle_Py.TraceThroughZOS(pth,surflist)

# Sweet now we have raydata, let's compute the differential ray transfer matrix
dMat = ComputeDifferentialFromRaybundles(raybundle_base,raybundle_Px,raybundle_Py,raybundle_Hx,raybundle_Hy)


import matplotlib.pyplot as plt
x = raybundle_base.xData[0]
y = raybundle_base.yData[1]
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

