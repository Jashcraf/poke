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
surflist = [2,3,6]
# surflist = [2,3]

# Thin Film Stuff
n_Ag = 0.12525718 - 1j*3.7249341450547577 # substrate
n_SiN = 2.00577335 # thin film
d = 0.0085e-6 #* 1e6


# t_Ag = 110e-9 #* 1e6
# n_ZD = 1.5418
# t_ZD = 50e-3 #* 1e6

n2 = [(n_SiN,d),(n_Ag)] # 
# n2 = n_Ag
# stack = [(n_SiN,d),(n_Ag,t_Ag)]
# stack = [(n_SiN,d),(n_Ag,t_Ag),(n_ZD,t_ZD)]
stack = None #[(n_SiN,d)]


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
th = np.pi/2


# try to run a raytrace
raybundle.TraceThroughZOS(pth,surflist)
raybundle.ConvertRayDataToPRTData()
raybundle.ComputePRTMatrix()
raybundle.ComputeTotalPRTMatrix()

# # What does the AOI look like?
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig,ax = plt.subplots(ncols=3,figsize=[12,3])
# for i,aoi_map in enumerate(raybundle.aoi):

#     im = ax[i].scatter(raybundle.xData[i],
#                        raybundle.yData[i],
#                        c=aoi_map)
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im,cax=cax)
#     ax[i].tick_params(labelbottom=False,labelleft=False)

# plt.show()

# fig,ax = plt.subplots(ncols=3,figsize=[12,3])
# plt.suptitle('|rs|')
# for i,aoi_map in enumerate(raybundle.aoi):

#     # compute the fresnel coefficients
#     rs,rp = pol.FresnelCoefficients(aoi_map,1,n2)

#     im = ax[i].scatter(raybundle.xData[i],
#                        raybundle.yData[i],
#                        c=np.abs(rs))
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im,cax=cax)
#     ax[i].tick_params(labelbottom=False,labelleft=False)

# plt.show()

# fig,ax = plt.subplots(ncols=3,figsize=[12,3])
# plt.suptitle('|rp|')
# for i,aoi_map in enumerate(raybundle.aoi):

#     # compute the fresnel coefficients
#     rs,rp = pol.FresnelCoefficients(aoi_map,1,n2)

#     im = ax[i].scatter(raybundle.xData[i],
#                        raybundle.yData[i],
#                        c=np.abs(rp))
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im,cax=cax)
#     ax[i].tick_params(labelbottom=False,labelleft=False)

# plt.show()

# fig,ax = plt.subplots(ncols=3,figsize=[12,3])
# plt.suptitle('Arg[rs]')
# for i,aoi_map in enumerate(raybundle.aoi):

#     # compute the fresnel coefficients
#     rs,rp = pol.FresnelCoefficients(aoi_map,1,n2)

#     im = ax[i].scatter(raybundle.xData[i],
#                        raybundle.yData[i],
#                        c=np.angle(rs))
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im,cax=cax)
#     ax[i].tick_params(labelbottom=False,labelleft=False)

# plt.show()

# fig,ax = plt.subplots(ncols=3,figsize=[12,3])
# plt.suptitle('Arg[rp]')
# for i,aoi_map in enumerate(raybundle.aoi):

#     # compute the fresnel coefficients
#     rs,rp = pol.FresnelCoefficients(aoi_map,1,n2)

#     im = ax[i].scatter(raybundle.xData[i],
#                        raybundle.yData[i],
#                        c=np.angle(rp))
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im,cax=cax)
#     ax[i].tick_params(labelbottom=False,labelleft=False)

# plt.show()



    

# plot.PRTPlot(raybundle,surf=1)
# plot.PRTPlot(raybundle,surf=2)
# plot.PRTPlot(raybundle,surf=3)
raybundle.PRTtoJonesMatrix(np.array([0.,-np.sin(th),np.cos(th)]),np.array([1.,0.,0.]))

plot.PlotJonesPupil(raybundle) # oops takes a while
# plot.PRTPlot(raybundle) # oops takes a while