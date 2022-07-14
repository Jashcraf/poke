import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plt

import sys
print(sys.executable)

# Initialize a Raybundle
nrays = 256
n1 = 1
n2 = 2.3669 + 1j*8.4177 # for subaru
# n2 = 0.958 + 6.690*1j
# pth = "C:/Users/jaren/Desktop/poke/Hubble_Test.zmx"
# surflist = [2,4,7]
pth = "C:/Users/jaren/Desktop/poke/Subaru_Telescope_nospider.zmx"
surflist = [2,3,5]
# pth = "C:/Users/jaren/Desktop/poke/chipman_ex12.40.zmx"
# surflist = [3,5]

# Initialize a ray bundle
raybundle = ray.RayBundle(nrays,n1,n2)
# try to run a raytrace
raybundle.TraceThroughZOS(pth,surflist)
raybundle.ConvertRayDataToPRTData()
raybundle.ComputePRTMatrix()
raybundle.ComputeTotalPRTMatrix()
raybundle.PRTtoJonesMatrix()

# What type are they
# print((raybundle.aoi[0]))
# print((raybundle.kin[0]))
# print((raybundle.kout[0]))

# ZOS Single Raytrace says the edge of the pupil has an AOI of ~6.2 degrees, does that track?
# plt.AOIPlot(raybundle)
# plt.JonesPlot(raybundle)
# plt.PRTPlot(raybundle,surf=0)
# plt.PRTPlot(raybundle,surf=1)
# plt.PRTPlot(raybundle,surf=2)
# plt.PRTPlot(raybundle)
# raybundle.WriteTotalPRTMatrix('Subaru_M1-M3_750nm.fits')
raybundle.WriteTotalJonesMatrix('Subaru_M1-M3_750nm.fits')
plt.PlotJonesPupil(raybundle)
# vmin_amp=0.9,vmax_amp=0.939,vmin_opd=-0.751,vmax_opd=-0.451
# plt.PlotRays(raybundle)








# %%
