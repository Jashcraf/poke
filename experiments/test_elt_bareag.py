import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot
import sys
from astropy.io import fits
print(sys.executable)

# Initialize a Raybundle
nrays = 128
pth = "C:/Users/LOFT_Olaf/Desktop/poke/ELT_Paper_1/ELT/ELT.zmx"
surflist = [1,3,5,8,12]

# source: Johnson and Christy 1972: n,k 0.188–1.94 µm
n_Ag = 0.052824 - 1j*3.4018
n_Si = 2.0543
d = 55e-9

# raytrace parameters
pupil_radius = 39146.4e-3/2 # m
max_fov = 1 # deg
observatory_pointing = 90 * np.pi/180

m1 = {
      'surf':surflist[0],
      'mode':'reflect',
      'coating':[(n_Si,d),n_Ag]
      }
      
m2 = {
      'surf':surflist[1],
      'mode':'reflect',
      'coating':[(n_Si,d),n_Ag]
      }
      
m3 = {
      'surf':surflist[2],
      'mode':'reflect',
      'coating':[(n_Si,d),n_Ag]
      }
      
k1 = {
      'surf':surflist[3],
      'mode':'reflect',
      'coating':[(n_Si,d),n_Ag]
      }
      
k2 = {
      'surf':surflist[4],
      'mode':'reflect',
      'coating':[(n_Si,d),n_Ag]
      }

surflist = [m1,m2,m3,k1,k2]
rays = pol.Rayfront(nrays,600*1e-9,pupil_radius,max_fov,circle=True)
rays.as_polarized(surflist)
rays.TraceRaysetZOS(pth)
rays.ComputeJonesPupil(aloc=-np.array([0,1,0]),exit_x=np.array([1.,0.,0.]))
rays.PlotPRTMatrix()
rays.PlotJonesPupil()
# rays.PlotPRTMatrix()