import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import poppy

save_pickle = True

# Initialize a Raybundle
nrays = 32
n1 = 1
n2 = 0.958 - 1j*6.69 # Al from text
radius = 8323.3e-3
ffov = 0.08
wlen = 550e-9
pth = "C:/Users/douglase/Desktop/poke/examples/validation/CLY12_44.zmx"

# Surface parameters are defined with Python dictionaries.
# 'surf' is the surface number in the raytrace lens data editor
# 'coating' is the thin film information. For a single layer, it's just the refractive index
s1 = {
    'surf':1,
    'coating':n2,
    'mode':'reflect'
}

s2 = {
    'surf':2,
    'coating':n2,
    'mode':'reflect'
}

# Initialize a Rayfront
raybundle = pol.Rayfront(nrays,wlen,radius,ffov,circle=False)
raybundle.as_polarized([s1,s2]) # pass the raybundle the surface list

# Trace the rays through a zemax optical system
raybundle.TraceRaysetZOS(pth,surfaces=[s1,s2])

# Save the raybundle as a pickle
if save_pickle:
    import pickle
    with open('examples/validation/CLY12_44_rays.pickle','wb') as f:
        pickle.dump(raybundle,f)

# Compute the Jones Pupil from the ZOS raytrace and coating data
raybundle.ComputeJonesPupil(aloc=np.array([0.,0.,1.]),exit_x=np.array([1.,0.,0.]))

# Now the plot is weird, but I'm fairly certain it uses real/imag
jpupil = raybundle.JonesPupil[0][...,0:2,0:2]
j00 = jpupil[...,0,0].reshape([nrays,nrays])
j01 = jpupil[...,0,1].reshape([nrays,nrays])
j10 = jpupil[...,1,0].reshape([nrays,nrays])
j11 = jpupil[...,1,1].reshape([nrays,nrays])*np.exp(1j*np.pi)

# Real Part
plt.figure()
plt.subplot(221)
plt.imshow(np.abs(j00),cmap='Oranges_r',vmin=0.9,vmax=0.939)
plt.colorbar()
plt.subplot(222)
plt.imshow(np.real(j01),cmap='Blues_r',vmin=-0.132,vmax=0.132)
plt.colorbar()
plt.subplot(223)
plt.imshow(np.real(j10),cmap='Blues_r',vmin=-0.132,vmax=0.132)
plt.colorbar()
plt.subplot(224)
plt.imshow(np.abs(j11),cmap='Oranges_r',vmin=0.9,vmax=0.939)
plt.colorbar()
#plt.show()

plt.figure()
plt.subplot(221)
plt.imshow(np.angle(j00),cmap='rainbow',vmin=-0.751,vmax=-0.451)
plt.colorbar()
plt.subplot(222)
plt.imshow(np.angle(j01),cmap='rainbow',vmin=0.831,vmax=0.847)
plt.colorbar()
plt.subplot(223)
plt.imshow(np.angle(j10),cmap='rainbow',vmin=0.831,vmax=0.847)
plt.colorbar()
plt.subplot(224)
plt.imshow(np.angle(j11),cmap='rainbow',vmin=-0.751,vmax=-0.451)
plt.colorbar()
plt.show()

