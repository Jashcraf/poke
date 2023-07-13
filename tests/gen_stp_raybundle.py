import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
from astropy.io import fits

pth = 'C:/Users/douglase/Desktop/stp_polarization/STP_TMA_nocoating.zmx'
pth_to_anche_jpupil = 'C:/Users/douglase/Desktop/stp_polarization/ramya_prt_data/'
nrays = 30
wave = 1
wavelength = 0.540e-6 # v band
epd = 6460e-3
hfov = .01
global_coords = True

# Bare aluminum
n_al_v = 0.97274+1j*6.5119
coating = n_al_v #[(n_al_v,110e-9),(n_al_v)]

# Build the Jones Pupil from Ramya
Jxx = fits.getdata(pth_to_anche_jpupil+'Exxr_V.fits') + 1j*fits.getdata(pth_to_anche_jpupil+'Exxi_V.fits')
Jxy = fits.getdata(pth_to_anche_jpupil+'Exyr_V.fits') + 1j*fits.getdata(pth_to_anche_jpupil+'Exyi_V.fits')
Jyx = fits.getdata(pth_to_anche_jpupil+'Eyxr_V.fits') + 1j*fits.getdata(pth_to_anche_jpupil+'Eyxi_V.fits')
Jyy = fits.getdata(pth_to_anche_jpupil+'Eyyr_V.fits') + 1j*fits.getdata(pth_to_anche_jpupil+'Eyyi_V.fits')

print('Jones shape = ',Jxx.shape)
nrays = Jxx.shape[0]


s1 = {
    'surf':4,
    'coating':coating,
    'mode':'reflect'
}

s2 = {
    'surf':5,
    'coating':coating,
    'mode':'reflect'
}

s3 = {
    'surf':6,
    'coating':coating,
    'mode':'reflect'
}

s4 = {
    'surf':8,
    'coating':coating,
    'mode':'reflect'
}

surflist = [s1,s2,s3,s4]

def GenSTPRayfront():

    raybundle = Rayfront(nrays,wavelength,epd/2,hfov,
                         normalized_pupil_radius=1,fov=[0.,0.],circle=False)
    raybundle.as_polarized(surflist)
    raybundle.TraceRaysetZOS(pth)
    return raybundle
    
if __name__ == '__main__':
    
    rayfront = GenSTPRayfront()
    
    a = np.array([0.0000000000,-0.2303295358,0.9731126887])
    # a = np.array([0.,0.,1.])
    rayfront.ComputeJonesPupil(aloc=a,exit_x=-np.array([1.,0.,0.]))
    # plot.JonesPupil(rayfront)
    
    # get the Jones pupil from the rayfront
    Jpupil = rayfront.JonesPupil[0][...,:2,:2]
    Jpupil = Jpupil.reshape([nrays,nrays,2,2])
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.angle(Jpupil[...,1,1]))
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.angle(Jyy))
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.angle(Jpupil[...,1,1])-np.angle(Jyy))
    plt.colorbar()
    plt.show()
