import numpy as np
import sys
sys.path.insert(1,'C:/Users/UASAL-OPTICS/Desktop/poke/') # One day I'll actually package poke!
import poke.poke_core as pol
from astropy.io import fits
import astropy.units as u
from hcipy import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time

# ELT Aperture
# Awesome, now generate a Jones pupil with 256 pixels
import poke.raytrace as rt
import poke.plotting as plot

# Initialize a Raybundle
nrays = 256
n1 = 1
pth = "C:/Users/UASAL-OPTICS/Desktop/poke/ELT_Paper_1/ELT/ELT.zmx"
surflist = [1,3,5,8,12]
# surflist = [2,3]
pth_to_box = "C:/Users/UASAL-OPTICS/Box/PolarizationGSMT/ELT/Jones_Pupils/"

# Load Refractive Indices
index_pth = "C:/Users/UASAL-OPTICS/Box/PolarizationGSMT/refractive_indices.csv"
data = np.genfromtxt(index_pth,delimiter=',',skip_header=2)[:,1:]
wlen_microns = data[:,0]
N_Ag = data[:,1] - 1j*data[:,2]
N_Al = data[:,3] - 1j*data[:,4]
N_SiN = data[:,5] - 1j*data[:,6]

# Film thickness for ELT
t_SiN = 55e-10 # 55 angstroms

filters = ['U','B','g','V','R','I','z','y','J','H','K','L','M','N']


t1 = time()
for i,wlen in enumerate(wlen_microns):

    #  Exxr_B.fits, Exxi_B.fits
    n2 = [(N_SiN[i],t_SiN),(N_Ag[i])] 

    raybundle = rt.Rayfront(nrays,n1,n2,wlen*1e-6,circle=False)
    raybundle.TraceThroughZOS(pth,surflist)
    raybundle.ConvertRayDataToPRTData()
    raybundle.ComputePRTMatrix()
    raybundle.ComputeTotalPRTMatrix()
    raybundle.PRTtoJonesMatrix(np.array([0.,-1.,0.]),np.array([1.,0.,0.]))

    # write with Ramya's convention
    Exx = raybundle.Jtot[0,0].reshape([256,256])
    Exy = raybundle.Jtot[0,1].reshape([256,256])
    Eyx = raybundle.Jtot[1,0].reshape([256,256])
    Eyy = raybundle.Jtot[1,1].reshape([256,256])

    # double them up and throw in fits files
    Exx_r = fits.HDUList([fits.PrimaryHDU(np.real(Exx))])
    Exx_i = fits.HDUList([fits.PrimaryHDU(np.imag(Exx))])

    Exy_r = fits.HDUList([fits.PrimaryHDU(np.real(Exy))])
    Exy_i = fits.HDUList([fits.PrimaryHDU(np.imag(Exy))])

    Eyx_r = fits.HDUList([fits.PrimaryHDU(np.real(Eyx))])
    Eyx_i = fits.HDUList([fits.PrimaryHDU(np.imag(Eyx))])

    Eyy_r = fits.HDUList([fits.PrimaryHDU(np.real(Eyy))])
    Eyy_i = fits.HDUList([fits.PrimaryHDU(np.imag(Eyy))])

    # Now write them
    Exx_r.writeto(pth_to_box+'Exxr_'+filters[i]+'.fits')
    Exx_i.writeto(pth_to_box+'Exxi_'+filters[i]+'.fits')

    Exy_r.writeto(pth_to_box+'Exyr_'+filters[i]+'.fits')
    Exy_i.writeto(pth_to_box+'Exyi_'+filters[i]+'.fits')

    Eyx_r.writeto(pth_to_box+'Eyxr_'+filters[i]+'.fits')
    Eyx_i.writeto(pth_to_box+'Eyxi_'+filters[i]+'.fits')
    
    Eyy_r.writeto(pth_to_box+'Eyyr_'+filters[i]+'.fits')
    Eyy_i.writeto(pth_to_box+'Eyyi_'+filters[i]+'.fits')

print('time to compute = ',time()-t1)
# from astropy.io import fits
# # grab AOI
# for i,surf_aoi in enumerate(raybundle.aoi):

#     reshaped_aoi = surf_aoi.reshape([nrays,nrays])
#     aoi_hdul = fits.HDUList([fits.PrimaryHDU(reshaped_aoi)])
#     aoi_hdul.writeto('C:/Users/UASAL-OPTICS/Desktop/poke/ELT_Paper_1/ELT/AOI_Dia_Ret/ELT_aoi_surf{}.fits'.format(i))



