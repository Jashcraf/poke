import poke.gbd as gbd
#import poke.worku_gbd as worku
import numpy as np
import poke.raytrace as ray

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

dplusses,dminuses = gbd.ComputeFinitePropagation(raybundle_base,1e-3)
print(dplusses[0])
print(dminuses[0])
print(raybundle_base.xData[-1][0])
print(raybundle_base.yData[-1][0])
print(raybundle_base.zData[-1][0])

# now update the positions to be at the minus detector plane
raybundle_base.MarchRayfront(dminuses)
raybundle_Px.MarchRayfront(dminuses)
raybundle_Py.MarchRayfront(dminuses)
raybundle_Hx.MarchRayfront(dminuses)
raybundle_Hy.MarchRayfront(dminuses)

# check the ray data
print(raybundle_base.xData[-1][0])
print(raybundle_base.yData[-1][0])
print(raybundle_base.zData[-1][0])

# March the ray data npix times
npix = 256
detsize = 1e-3
dy = detsize/npix








