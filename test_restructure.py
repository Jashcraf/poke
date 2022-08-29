import poke.poke_core as poke

# test init a raybundle
nrays = 51
wavelength = 1.65e-6
pupil_radius = 1.2
max_fov = 0.08

pth = "C:/Users/UASAL-OPTICS/Desktop/poke/Hubble_Test.zmx"
surflist = [2,4,7]


s1 = {
    'surf' : 2,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

s2 = {
    'surf' : 4,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

s3 = {
    'surf' : 7,
    'mode' : 'reflect',
    'coating': 2.3669 + 1j*8.4177 # for subaru
}

surfaces = [s1,s2,s3]

# Instantiate Class
rays = poke.Rayfront(nrays,wavelength,pupil_radius,max_fov)

# This should be a @classmethod but I don't get how they work yet
rays.as_polarized(surfaces)

# Now trace the rays
rays.TraceRaysetZOS(pth)

# And display them, surface number is the *index* in surfaces list
rays.PlotRaysAtSurface(2)