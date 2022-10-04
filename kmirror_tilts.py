import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot
import sys
from astropy.io import fits
print(sys.executable)

def WriteFits(array,pth):
    to_write = fits.HDUList([fits.PrimaryHDU(array)])
    to_write.writeto(pth,overwrite=True)
    
# Initialize a Raybundle
nrays = 32
pth = "C:/Users/LOFT_Olaf/Desktop/poke/imrotate_vs_wlen_test/subaru_observatory_45_imrotate_90.zmx"
pth_jones = "C:/Users/LOFT_Olaf/Desktop/poke/imrotate_vs_wlen_test/Jones_Pupils/"
surflist = [1,2,4,10,13,16]

wlen = np.arange(600,800,25)

# source: Johnson and Christy 1972: n,k 0.188–1.94 µm
n_Ag = [0.055159-1j*4.0097, 0.058080-1j*4.2156, 0.052225-1j*4.4094,0.046556-1j*4.6053,
        0.041000-1j*4.8025, 0.036019-1j*4.9988, 0.031165-1j*5.1949,0.032919-1j*5.3836,
        0.036759-1j*5.5698]
        
# source: Cheng et al. 2016: n,k 0.225–1.00 µm
n_Al = [0.73082-1j*5.7568, 0.81663-1j*5.9920, 0.91625-1j*6.2210, 1.0327-1j*6.4397,
        1.1693-1j*6.6417, 1.3280-1j*6.8156, 1.5061-1j*6.9439, 1.6893-1j*7.0044,
        1.8385-1j*6.9757]
        
# source: Malitson 1965: n 0.21–3.71 µm
n_SiO2 = [1.4580,1.4572,1.4565,1.4559,
          1.4553,1.4547,1.4542,1.4538,
          1.4533]
          
d = 262.56e-9/3 # effective film thickness

# raytrace parameters
pupil_radius = 4.1 # m
max_fov = 1 # deg
observatory_pointing = 45 * np.pi/180

for i,wlen in enumerate(wlen):

    m1 = {
          'surf':1,
          'mode':'reflect',
          'coating':n_Al[i]
          }
          
    m2 = {
          'surf':2,
          'mode':'reflect',
          'coating':n_Al[i]
          }
          
    m3 = {
          'surf':4,
          'mode':'reflect',
          'coating':n_Al[i]
          }
          
    k1 = {
          'surf':10,
          'mode':'reflect',
          'coating':[(n_SiO2[i],d),n_Ag[i]]
          }
          
    k2 = {
          'surf':13,
          'mode':'reflect',
          'coating':[(n_SiO2[i],d),n_Ag[i]]
          }
          
    k3 = {
          'surf':16,
          'mode':'reflect',
          'coating':[(n_SiO2[i],d),n_Ag[i]]
          }

    surflist = [m1,m2,m3,k1,k2,k3]
    rays = pol.Rayfront(nrays,wlen*1e-9,pupil_radius,max_fov,circle=False)
    rays.as_polarized(surflist)
    rays.TraceRaysetZOS(pth)
    rays.ComputeJonesPupil(aloc=-np.array([0,-np.sin(observatory_pointing),np.cos(observatory_pointing)]),exit_x=np.array([1.,0.,0.]))
    # rays.PlotPRTMatrix()
    # rays.PlotJonesPupil()
    J = rays.JonesPupil[0]
    
    Axx = np.reshape(np.abs(J[:,0,0]),[nrays,nrays])
    Axy = np.reshape(np.abs(J[:,0,1]),[nrays,nrays])
    Ayx = np.reshape(np.abs(J[:,1,0]),[nrays,nrays])
    Ayy = np.reshape(np.abs(J[:,1,1]),[nrays,nrays])
    
    Pxx = np.reshape(np.abs(J[:,0,0]),[nrays,nrays])
    Pxy = np.reshape(np.abs(J[:,0,1]),[nrays,nrays])
    Pyx = np.reshape(np.abs(J[:,1,0]),[nrays,nrays])
    Pyy = np.reshape(np.abs(J[:,1,1]),[nrays,nrays])
    
    
    WriteFits(Axx,pth_jones+'Axx_{}_90.fits'.format(wlen))
    WriteFits(Axy,pth_jones+'Axy_{}_90.fits'.format(wlen))
    WriteFits(Ayx,pth_jones+'Ayx_{}_90.fits'.format(wlen))
    WriteFits(Ayy,pth_jones+'Ayy_{}_90.fits'.format(wlen))
    
    WriteFits(Pxx,pth_jones+'Pxx_{}_90.fits'.format(wlen))
    WriteFits(Pxy,pth_jones+'Pxy_{}_90.fits'.format(wlen))
    WriteFits(Pyx,pth_jones+'Pyx_{}_90.fits'.format(wlen))
    WriteFits(Pyy,pth_jones+'Pyy_{}_90.fits'.format(wlen))