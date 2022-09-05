import poke.poke_core as poke
import numpy as np

# test init a raybundle
nrays = 51
wavelength = 1.65e-6
pupil_radius = 1.2
max_fov = 0.08

pth = "C:/Users/UASAL-OPTICS/Desktop/poke/Subaru_Telescope_nospider.zmx"


s1 = {
    'surf' : 2,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

s2 = {
    'surf' : 3,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

s3 = {
    'surf' : 5,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

surfaces = [s1,s2]

# Instantiate Class
rays = poke.Rayfront(nrays,wavelength,pupil_radius,max_fov)

# This should be a @classmethod but I don't get how they work yet
rays.as_polarized(surfaces)

# Now trace the rays
rays.TraceRaysetZOS(pth)

# And display them, surface number is the *index* in surfaces list
# rays.PlotRaysAtSurface(0)
# rays.PlotRaysAtSurface(1)

# Create & Plot Jones Pupil
rays.ComputeJonesPupil(aloc=np.array([0.,0.,1.]),exit_x=np.array([1,0,0]))
rays.PlotJonesPupil()
rays.PlotPRTMatrix()