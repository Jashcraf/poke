import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray

# Initialize a Raybundle
nrays = 25
n1 = 1
n2 = 2.3669 + 1j*8.4177
pth = "C:/Users/jaren/Desktop/poke/Hubble_Test.zmx"

# Initialize a ray bundle
raybundle = ray.RayBundle(nrays,n1,n2)

# try to run a raytrace
raybundle.TraceThroughZOS(pth,[2])
raybundle.ConvertRayDataToPRTData()

print(raybundle.kin)
