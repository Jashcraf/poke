import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot

import sys
print(sys.executable)

# Initialize a Raybundle
nrays = 32
n1 = 1
n2 = 2.3669 + 1j*8.4177 # bare Aluminum at 750nm for subaru
# n2 = 0.031165 + 1j*5.1949 # bare silver at 750nm for subaru
# n2 = 1.4920 + 1j*14.799 # bare Aluminum at 1.471um
pth = "C:/Users/jaren/Desktop/poke/subaru_imrotates/subaru_imrotate_"
fn_ini = "derotator_jpupil_750nm_Al_"
surflist = [1,2,4,10,13,16]

angles = np.arange(0,100,10)

for angle in angles:

    aloc = np.array([0.,0.,1.])
    exit_x = np.array([1.,0.,0.])

    to_load = pth+str(int(angle))+".zmx"

    # Initialize a ray bundle
    raybundle = ray.RayBundle(nrays,n1,n2)

    # try to run a raytrace
    raybundle.TraceThroughZOS(to_load,surflist)
    raybundle.ConvertRayDataToPRTData()
    raybundle.ComputePRTMatrix()
    raybundle.ComputeTotalPRTMatrix()
    raybundle.PRTtoJonesMatrix(aloc,exit_x)
    plot.PlotJonesPupil(raybundle) # oops takes a while

    fn = fn_ini+str(int(angle))+".fits"

    raybundle.WriteTotalJonesMatrix(fn)



