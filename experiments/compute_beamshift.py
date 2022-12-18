import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import zoom, center_of_mass
from matplotlib.colors import LogNorm
from prysm.propagation import focus_fixed_sampling

# Subaru Params
nrays = 32
dia = 8.2 # meters
input_dx = dia/256 # meters
prop_dist = 110.629 # meters
ref_wlen = 600e-9 # meters
output_dx = 1.22*ref_wlen*prop_dist/dia/16 # sample the airy radius 4 times
output_samples = 256


def PropToImage(array):
    return np.fft.fftshift(np.fft.fft2(array,norm='ortho'))

pth_to_jones = "imrotate_vs_wlen_test/Jones_Pupils/"
wlens = np.arange(600,825,25)
angles = np.arange(0,100,10)
beam_shift_x = []
beam_shift_y = []
# angle = 90

beamshift_box = np.empty([2,angles.shape[0],wlens.shape[0]])

for i,angle in enumerate(angles):

    for j,wlen in enumerate(wlens):

        # Load complex E fields
        Exx = fits.open(pth_to_jones+"Axx_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pxx_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data)
        Exy = fits.open(pth_to_jones+"Axy_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pxy_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data)
        Eyx = fits.open(pth_to_jones+"Ayx_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pyx_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data)
        Eyy = fits.open(pth_to_jones+"Ayy_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pyy_{wl}_{ang}.fits".format(wl=wlen,ang=angle))[0].data)
        
        wlen *= 1e-9
        print(wlen)

        # Interpolate because we can
        Exx = zoom(Exx,256/32)
        Exy = zoom(Exy,256/32)
        Eyx = zoom(Eyx,256/32)
        Eyy = zoom(Eyy,256/32)

        # compute pixelscale
        # pixscale = 8.2/256 # m/pix

        x = np.linspace(-1,1,Exx.shape[0])
        x,y = np.meshgrid(x,x)
        mask = np.zeros(Exx.shape)
        mask[x**2 + y**2 <= 1] = 1

        Exx *= mask
        Exy *= mask
        Eyx *= mask
        Eyy *= mask

        # Plot to make sure they loaded right
        # plt.figure(figsize=[10,5])
        # plt.subplot(241)
        # plt.imshow(np.abs(Exx/mask),cmap='inferno')
        # plt.colorbar()
        # plt.title('Axx')

        # plt.subplot(242)
        # plt.imshow(np.abs(Exy/mask),cmap='inferno')
        # plt.colorbar()
        # plt.title('Axy')

        # plt.subplot(245)
        # plt.imshow(np.abs(Eyx/mask),cmap='inferno')
        # plt.colorbar()
        # plt.title('Ayx')

        # plt.subplot(246)
        # plt.imshow(np.abs(Eyy/mask),cmap='inferno')
        # plt.colorbar()
        # plt.title('Ayy')

        # plt.subplot(243)
        # plt.imshow(np.angle(Exx/mask),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Axx')

        # plt.subplot(244)
        # plt.imshow(np.angle(Exy/mask),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Axy')

        # plt.subplot(247)
        # plt.imshow(np.angle(Eyx/mask),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Ayx')

        # plt.subplot(248)
        # plt.imshow(np.angle(Eyy/mask),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Ayy')
        # plt.show()
        
        # print('propagation parameters')
        # print('----------------------')
        # print('dx in = ',input_dx)
        # print('efl = ',prop_dist)
        # print('dx out = ',output_dx)

        # Now Zero-pad the arrays by some oversample and take the FFT
        Axx = focus_fixed_sampling(Exx,input_dx,prop_dist,wlen,output_dx,output_samples) #PropToImage(Exx)
        Axy = focus_fixed_sampling(Exy,input_dx,prop_dist,wlen,output_dx,output_samples) #PropToImage(Exy)
        Ayx = focus_fixed_sampling(Eyx,input_dx,prop_dist,wlen,output_dx,output_samples) #PropToImage(Eyx)
        Ayy = focus_fixed_sampling(Eyy,input_dx,prop_dist,wlen,output_dx,output_samples) #PropToImage(Eyy)

        # Plot to make sure they loaded right
        # plt.figure(figsize=[10,5])
        # plt.subplot(241)
        # plt.imshow(np.abs(Axx),cmap='inferno',norm=LogNorm())
        # plt.colorbar()
        # plt.title('Axx')

        # plt.subplot(242)
        # plt.imshow(np.abs(Axy),cmap='inferno',norm=LogNorm())
        # plt.colorbar()
        # plt.title('Axy')

        # plt.subplot(245)
        # plt.imshow(np.abs(Ayx),cmap='inferno',norm=LogNorm())
        # plt.colorbar()
        # plt.title('Ayx')

        # plt.subplot(246)
        # plt.imshow(np.abs(Ayy),cmap='inferno',norm=LogNorm())
        # plt.colorbar()
        # plt.title('Ayy')

        # plt.subplot(243)
        # plt.imshow(np.angle(Axx),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Axx')

        # plt.subplot(244)
        # plt.imshow(np.angle(Axy),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Axy')

        # plt.subplot(247)
        # plt.imshow(np.angle(Ayx),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Ayx')

        # plt.subplot(248)
        # plt.imshow(np.angle(Ayy),cmap='coolwarm')
        # plt.colorbar()
        # plt.title('Ayy')
        # plt.show()

        # Print Out the Difference in Beamshift in pixels
        com_x = np.array(center_of_mass(np.abs(Axx+Axy)**2))
        com_y = np.array(center_of_mass(np.abs(Ayy+Ayx)**2))
        
        # Revise Pixel Scale
        # fft frequencies 
        # y,x = Exx.shape
        # y_freq = np.fft.fftshift(np.fft.fftfreq(y,pixscale))
        # x_freq = np.fft.fftshift(np.fft.fftfreq(x,pixscale))
        # new_pixscale = x_freq
        
        #new_pixscale = pixscale/os # m/pixel, before propagation
        #new_pixscale = (wlen*1e-9/8.2)/os
        #new_pixscale /= wlen*1e-9 * 1.106292158265238e2  # m^-1/pixel, convert to spatial frequency units
        #new_pixscale *= wlen*1e-9 # focal length of subaru; rad/pixel
        #new_pixscale *= 206265 # convert to arcsec; as pixel 
        new_pixscale = np.arctan(output_dx/prop_dist)*206265
        # print('new pixelscale [mas]',new_pixscale*1e3)

        print('Com X-polarization [pix] = ',com_x)
        print('Com Y-polarization [pix] = ',com_y)
        print('total beamshift in y and x [mas] = ',(com_x-com_y)*new_pixscale*1e3)
        beamshift_box[0,i,j] = ((com_x-com_y)*new_pixscale)[0]*1e3
        beamshift_box[1,i,j] = ((com_x-com_y)*new_pixscale)[1]*1e3
        # beam_shift_x.append() # converts to mas
        # beam_shift_y.append(((com_x-com_y)*new_pixscale)[1]*1e3) # converts to mas

plt.figure(figsize=[10,5])
plt.subplot(121)
plt.title('Differential Beam Shift in X [mas]')
plt.imshow(beamshift_box[0],interpolation='None',vmax=1.5,vmin=-1.5)
plt.xticks(ticks = np.arange(0,9,1),labels=wlens)
plt.yticks(ticks=np.arange(0,10,1),labels=angles)
plt.xlabel('Wavelength [nm]')
plt.ylabel(' K mirror angle w.r.t. yz-plane [deg]')
plt.colorbar()
plt.subplot(122)
plt.title('Differential Beam Shift in Y [mas]')
plt.imshow(beamshift_box[1],interpolation='None',vmax=1.5,vmin=-1.5)
plt.xticks(ticks = np.arange(0,9,1),labels=wlens)
plt.yticks(ticks=np.arange(0,10,1),labels=angles)
plt.xlabel('Wavelength [nm]')
plt.colorbar()
plt.show()

magnitude = np.sqrt(beamshift_box[0]**2 + beamshift_box[1]**2)
angle = np.arctan2(beamshift_box[1],beamshift_box[0])
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.title('Magnitude of Differential Beam Shift [mas]')
plt.imshow(magnitude,interpolation='None')
plt.xticks(ticks = np.arange(0,9,1),labels=wlens)
plt.yticks(ticks=np.arange(0,10,1),labels=angles)
plt.xlabel('Wavelength [nm]')
plt.ylabel(' K mirror angle w.r.t. yz-plane [deg]')
plt.colorbar()
plt.subplot(122)
plt.title('Angle of Differential Beam Shift [mas]')
plt.imshow(angle,interpolation='None')
plt.xticks(ticks = np.arange(0,9,1),labels=wlens)
plt.yticks(ticks=np.arange(0,10,1),labels=angles)
plt.xlabel('Wavelength [nm]')
plt.colorbar()
plt.show()


# plt.figure()
# plt.plot(wlens,beam_shift_x,label='x')
# plt.plot(wlens,beam_shift_y,label='y')
# plt.xlabel('Wavelength [nm]')
# plt.ylabel('cen[Ex] - cen{Ey} [mas]')
# plt.legend()
# plt.show()
# plt.figure()
# plt.imshow(np.log10(np.abs(np.abs(Axx+Axy)**2)),vmax=2,vmin=-10)
# plt.xlim([npix-cut,npix+cut])
# plt.ylim([npix-cut,npix+cut])
# plt.colorbar()
# plt.title('Exx')
# plt.show()

# plt.figure()
# plt.imshow(np.log10(np.abs(np.abs(Ayy+Ayx)**2)),vmax=2,vmin=-10)
# plt.xlim([npix-cut,npix+cut])
# plt.ylim([npix-cut,npix+cut])
# plt.colorbar()
# plt.title('Eyy')
# plt.show()

# plt.figure()
# plt.imshow(np.log10(np.abs(np.abs(Axx+Axy)**2-np.abs(Ayy+Ayx)**2)),vmax=2,vmin=-10)
# plt.xlim([npix-cut,npix+cut])
# plt.ylim([npix-cut,npix+cut])
# plt.colorbar()
# plt.title('Image Difference')
# plt.show()
