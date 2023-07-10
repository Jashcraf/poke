import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import poppy

save_pickle = True

# Initialize a Raybundle
nrays = 256
n1 = 1
n2 = 1.0194 - 1j*6.6388 # Al in v band
radius = 1.2
ffov = 0.08
wlen = 551e-9
pth = "C:/Users/douglase/Desktop/poke/test_files/Hubble_Test.zmx"

# Surface parameters are defined with Python dictionaries.
# 'surf' is the surface number in the raytrace lens data editor
# 'coating' is the thin film information. For a single layer, it's just the refractive index
s1 = {
    'surf':2,
    'coating':n2,
    'mode':'reflect'
}

s2 = {
    'surf':4,
    'coating':n2,
    'mode':'reflect'
}

# Initialize a Rayfront
raybundle = pol.Rayfront(nrays,wlen,radius,ffov,circle=False)
raybundle.as_polarized([s1,s2]) # pass the raybundle the surface list

# Trace the rays through a zemax optical system
raybundle.TraceRaysetZOS(pth,surfaces=[s1,s2])

# Compute the Jones Pupil from the ZOS raytrace and coating data
# raybundle.ComputeJonesPupil(aloc=np.array([0.,0.,1.]),exit_x=np.array([1.,0.,0.]))

# Now plot the Jones Pupil
# plot.JonesPupil(raybundle)

# Compute the ARM
# ARM = raybundle.ComputeARM(pad=8)

# Plot the ARM
# plot.AmplitudeResponseMatrix(ARM,lim=128)

# Compute the PSM
# P00 = raybundle.ComputePSM(stokes=np.array([1.,0.,0.,0.]),cut=32)

# plot.PointSpreadMatrix(raybundle.PSM)

print(raybundle.opd.shape)
plot.RayOPD(raybundle)

