import numpy as np
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
from poke.poke_core import Rayfront
from poke.writing import write_rayfront_to_serial,read_serial_to_rayfront
import pickle
from astropy.io import fits

print(globals()['Rayfront'])

pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/experiments/ELT_Paper_1/ELT/ELT_551.zmx'
nrays = 64
wave = 1
global_coords = True

n_Al = 1.2 + 1j*7.115 # 600nm from CV Al coating MUL
n_Al = n_Al #np.complex64(n_Al)

s1 = {
    'surf':1,
    'coating':n_Al,
    'mode':'reflect'
}

s2 = {
    'surf':3,
    'coating':n_Al,
    'mode':'reflect'
}

s3 = {
    'surf':5,
    'coating':n_Al,
    'mode':'reflect'
}

s4 = {
    'surf':8,
    'coating':n_Al,
    'mode':'reflect'
}

s5 = {
    'surf':12,
    'coating':n_Al,
    'mode':'reflect'
}

si = {
    'surf':11,
    'coating':1,
    'mode':'reflect'
}

surflist = [s1,s2,s3,s4,s5]

# def trace_beamlet_rayfront():

#     wavelength = 1.65
#     pupil_radius = 1.2
#     max_fov = 0.08
#     OF = 1.4
#     wo = 2*pupil_radius*OF / (2*nrays)
#     dsize = 0.5e-3
#     npix = 128
#     x = np.linspace(-dsize/2,dsize/2,npix)
#     x,y = np.meshgrid(x,x)
#     dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])

#     surflist = [s1,si]

#     rf = Rayfront(nrays,wavelength*1e-6,pupil_radius,max_fov,waist_pad=wo)
#     rf.as_gaussianbeamlets(wo)
#     rf.trace_rayset(pth,surfaces=surflist)
#     # field = rf.beamlet_decomposition_field(dcoords).reshape([npix,npix])

#     # plt.figure()
#     # plt.imshow(np.abs(field)**2)
#     # plt.colorbar()
#     # plt.show()

#     # now save
#     write_rayfront_to_serial(rf,f'hst_rayfront_asbeamlets_{nrays}rays_{wavelength}um')

def trace_polarized_rayfront():

    wavelength = 0.6
    surflist = [s1,s2,s3,s4,s5]
    pupil_radius = 39146.4/2
    max_fov = .0833
    rf = Rayfront(nrays,wavelength*1e-6,pupil_radius,max_fov)
    rf.as_polarized(surflist)
    rf.trace_rayset(pth)
    # rf.compute_jones_pupil(aloc=np.array([0.,1.,0.]))
    # plot.jones_pupil(rf)

    # save
    write_rayfront_to_serial(rf,f'ELT_rayfront_aspolarized_{nrays}rays_{wavelength}um')

def read_rayfronts():

    # pth_beams = 'hst_rayfront_asbeamlets_32rays_1.65um.msgpack'
    pth_polar = 'ELT_rayfront_aspolarized_64rays_0.6um.msgpack'

    # rf_beams = read_serial_to_rayfront(pth_beams)
    rf_polar = read_serial_to_rayfront(pth_polar)

    # Try the beamlet calculation
    # set up detector coordinates
    # dsize = 0.5e-3
    # npix = 128
    # x = np.linspace(-dsize/2,dsize/2,npix)
    # x,y = np.meshgrid(x,x)
    # dcoords = np.asarray([x.ravel(),y.ravel(),0*x.ravel()])

    # field = rf_beams.beamlet_decomposition_field(dcoords).reshape([npix,npix])
    # plt.figure()
    # plt.imshow(np.log10(np.abs(field)**2))
    # plt.colorbar()
    # plt.show()

    # now try the jones pupil
    rf_polar.compute_jones_pupil(aloc=np.array([0.,1.,0.]),exit_x=np.array([1.,0.,0.]))
    plot.jones_pupil(rf_polar)

if __name__ == '__main__':

    # trace_beamlet_rayfront()
    trace_polarized_rayfront()
    # read_rayfronts()