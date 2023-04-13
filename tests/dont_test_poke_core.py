# Tests the core poke module that holds the raytracer
import poke.poke_core as poke

# TODO: When CODE V becomes supported we will need another copy of tests
testfile = 'Hubble_Test.zmx'
nrays = 11
wavelength = 1e-6
pupil_radius = 1.2
max_fov = 0.008

# refractiveindex.info Johnson and Christy 1972: n,k 0.188–1.94 µm
n_Ag = 0.040000 - 1j*7.1155


s1 = {'surf':2,
      'coating':n_Ag,
      'mode':'reflect'
      }

s2 = {'surf':3,
      'coating':n_Ag,
      'mode':'reflect'
      }

def test_rayfront_raytrace():

    # Initialize the Rayfront
    raybundle = poke.Rayfront(nrays,wavelength,pupil_radius,max_fov)
    raybundle.as_polarized([s1,s2])
    raybundle.TraceRaysetZOS(testfile)
    raybundle.ComputeJonesPupil()

    pass

def test_rayfront_aspolarized():
    pass

def test_rayfront_asgausslets():
    pass