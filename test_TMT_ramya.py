import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot

import sys

# Initialize a Raybundle
nrays = 64
n1 = 1
# n2 = 2.3669 + 1j*8.4177 # bare Aluminum at 750nm for subaru
# n2 = 1.4920 + 1j*14.799 # bare Aluminum at 1.471um
pth = "C:/Users/UASAL-OPTICS/Desktop/poke/TMT_non-segmented.zmx"
surflist = [2,3,5]
# surflist = [2,3]

# Thin Film Stuff
n_Ag = 0.12525718 - 1j*3.7249341450547577 
n_SiN = 2.00577335
t_Ag = 110e-9 #* 1e6
n_ZD = 1.5418
t_ZD = 50e-3 #* 1e6
d = 0.0085e-6 #* 1e6

n2 = n_Ag # 
stack = [(n_SiN,d),(n_Ag,t_Ag)]
stack = [(n_SiN,d)]

import matplotlib.pyplot as plt

# aoi = np.linspace(0,6,200) 
# rtot_s = pol.HartenTwoLayerFilm(aoi* np.pi/180,n_SiN,d,n_Ag,'s')
# rtot_p = pol.HartenTwoLayerFilm(aoi* np.pi/180,n_SiN,d,n_Ag,'p')

# plt.figure()
# plt.title('Harten Effective r coefficient v.s. aoi')
# plt.plot(aoi,np.abs(rtot_s),label='|s-pol|')
# # plt.plot(aoi,np.angle(rtot_s),label='Arg{s-pol}')
# plt.plot(aoi,np.abs(rtot_p),label='|p-pol|')
# # plt.plot(aoi,np.angle(rtot_p),label='Arg{p-pol}')
# plt.xlabel('AOI [deg]')
# plt.ylabel('Fresnel Coefficient Value')
# plt.legend()
# plt.show()

# Initialize a ray bundle
raybundle = ray.Rayfront(nrays,n1,n2,.6e-6,stack=stack)

# try to run a raytrace
raybundle.TraceThroughZOS(pth,surflist)
raybundle.ConvertRayDataToPRTData()
raybundle.ComputePRTMatrix()
raybundle.ComputeTotalPRTMatrix()
raybundle.PRTtoJonesMatrix(np.array([0.,-1.,0.]),np.array([1.,0.,0.]))

plot.PlotJonesPupil(raybundle) # oops takes a while
plot.PRTPlot(raybundle) # oops takes a while