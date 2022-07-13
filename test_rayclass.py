import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plt

# Initialize a Raybundle
nrays = 25
n1 = 1
n2 = 2.3669 + 1j*8.4177
pth = "C:/Users/jaren/Desktop/poke/Hubble_Test.zmx"

# Initialize a ray bundle
raybundle = ray.RayBundle(nrays,n1,n2)

# try to run a raytrace
raybundle.TraceThroughZOS(pth,[7])
raybundle.ConvertRayDataToPRTData()

# What type are they
# print((raybundle.aoi[0]))
# print((raybundle.kin[0]))
# print((raybundle.kout[0]))

# ZOS Single Raytrace says the edge of the pupil has an AOI of ~12 degrees, does that track?
plt.AOIPlot(raybundle)



