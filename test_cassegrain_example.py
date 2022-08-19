# Run Example from PL&OS Section 12.5
# Goal is to generate the jones pupil from figure 12.44

import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot
import matplotlib.pyplot as plt
from astropy.io import fits

n1 = 1
n2 = 0.958 + 1j*6.690

# Initialize a Raybundle
nrays_across_pupil = 32
aloc = np.array([0.,0.,1.]) # antipole location
exit_x = np.array([1.,0.,0.]) # reference x axis
pth = "C:/Users/LOFT_Olaf/Desktop/poke/test_files/PL&OS_CassegrainJonesPupil.zmx"
surflist = [1,2]

# Initialize a ray bundle
raybundle = ray.Rayfront(nrays_across_pupil,n1,n2)

# run a raytrace (please forgive the API I promise I'm doing my best)
raybundle.TraceThroughZOS(pth,surflist)
raybundle.ConvertRayDataToPRTData()
raybundle.ComputePRTMatrix()
raybundle.ComputeTotalPRTMatrix()
raybundle.PRTtoJonesMatrix(aloc,exit_x)
plot.PlotJonesPupil(raybundle)

# raybundle.WriteTotalJonesMatrix('test_files/Cassegrain_JonesPupil_Test.fits')

# Now we load it 
j = fits.open('test_files/Cassegrain_JonesPupil_Test.fits')[0].data
j00 = j[:,:,0,0,0] + 1j*j[:,:,0,0,1]
j01 = j[:,:,0,1,0] + 1j*j[:,:,0,1,1]
j10 = j[:,:,1,0,0] + 1j*j[:,:,1,0,1]
j11 = j[:,:,1,1,0] + 1j*j[:,:,1,1,1]

# Set up the big plot

plt.figure(figsize=[12,7])


# The Amplitudes
plt.subplot(241)
plt.imshow(np.abs(j00),vmin=0.9,vmax=0.939,cmap='gist_heat')
plt.colorbar()

plt.subplot(242)
plt.imshow(np.abs(j01),vmin=-0.132,vmax=0.132,cmap='bone')
plt.colorbar()

plt.subplot(245)
plt.imshow(np.abs(j10),vmin=-0.132,vmax=0.132,cmap='bone')
plt.colorbar()

plt.subplot(246)
plt.imshow(np.abs(j11),vmin=0.9,vmax=0.939,cmap='gist_heat')
plt.colorbar()

# The Phases
plt.subplot(243)
#plt.imshow(np.angle(j00),vmin=-0.751,vmax=-0.451,cmap='jet')
plt.imshow(np.angle(j00),cmap='jet')
plt.colorbar()

plt.subplot(244)
#plt.imshow(np.angle(j01),vmin=0.831,vmax=0.847,cmap='jet')
plt.imshow(np.angle(j01),cmap='jet')
plt.colorbar()

plt.subplot(247)
# plt.imshow(np.angle(j10),vmin=0.831,vmax=0.847,cmap='jet')
plt.imshow(np.angle(j10),cmap='jet')
plt.colorbar()

plt.subplot(248)
# plt.imshow(np.angle(j11),vmin=-0.751,vmax=-0.451,cmap='jet')
plt.imshow(np.angle(j11)+np.pi,cmap='jet')
plt.colorbar()

plt.show()
