import poke.poke_core as pol
import poke.raytrace as ray

# Initialize a Raybundle
nrays = 128
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
    'coating':n2
}

s2 = {
    'surf':4,
    'coating':n2
}

raybundle = pol.Rayfront(nrays,wlen,radius,ffov,circle=True)
raybundle.as_polarized([s1,s2]) # pass the raybundle the surface list
raybundle.TraceRaysetZOS(pth,surfaces=[s1,s2])