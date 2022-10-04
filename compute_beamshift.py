import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import zoom, center_of_mass
from matplotlib.colors import LogNorm


def PropToImage(array):
    return np.fft.fftshift(np.fft.fft2(array,norm='ortho'))

pth_to_jones = "imrotate_vs_wlen_test/Jones_Pupils/"
wlen = 600

# Load complex E fields
Exx = fits.open(pth_to_jones+"Axx_{}_90.fits".format(wlen))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pxx_{}_90.fits".format(wlen))[0].data)
Exy = fits.open(pth_to_jones+"Axy_{}_90.fits".format(wlen))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pxy_{}_90.fits".format(wlen))[0].data)
Eyx = fits.open(pth_to_jones+"Ayx_{}_90.fits".format(wlen))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pyx_{}_90.fits".format(wlen))[0].data)
Eyy = fits.open(pth_to_jones+"Ayy_{}_90.fits".format(wlen))[0].data * np.exp(-1j*fits.open(pth_to_jones+"Pyy_{}_90.fits".format(wlen))[0].data)

# Interpolate because we can
Exx = zoom(Exx,256/32)
Exy = zoom(Exy,256/32)
Eyx = zoom(Eyx,256/32)
Eyy = zoom(Eyy,256/32)

# compute pixelscale
pixscale = 8.2/256 # m/pix

x = np.linspace(-1,1,Exx.shape[0])
x,y = np.meshgrid(x,x)
mask = np.zeros(Exx.shape)
mask[x**2 + y**2 <= 1] = 1

Exx *= mask
Exy *= mask
Eyx *= mask
Eyy *= mask

# Plot to make sure they loaded right
plt.figure(figsize=[10,5])
plt.subplot(241)
plt.imshow(np.abs(Exx),cmap='inferno')
plt.colorbar()
plt.title('Axx')

plt.subplot(242)
plt.imshow(np.abs(Exy),cmap='inferno')
plt.colorbar()
plt.title('Axy')

plt.subplot(245)
plt.imshow(np.abs(Eyx),cmap='inferno')
plt.colorbar()
plt.title('Ayx')

plt.subplot(246)
plt.imshow(np.abs(Eyy),cmap='inferno')
plt.colorbar()
plt.title('Ayy')

plt.subplot(243)
plt.imshow(np.angle(Exx),cmap='coolwarm')
plt.colorbar()
plt.title('Axx')

plt.subplot(244)
plt.imshow(np.angle(Exy),cmap='coolwarm')
plt.colorbar()
plt.title('Axy')

plt.subplot(247)
plt.imshow(np.angle(Eyx),cmap='coolwarm')
plt.colorbar()
plt.title('Ayx')

plt.subplot(248)
plt.imshow(np.angle(Eyy),cmap='coolwarm')
plt.colorbar()
plt.title('Ayy')
plt.show()

# Now Zero-pad the arrays by some oversample and take the FFT
os = 4
cut = 64
npix = int(os*Exx.shape[0]/2)
dim = int(Exx.shape[0]/2)

# Revise Pixel Scale
new_pixscale = pixscale # m/pixel, before propagation
new_pixscale *= wlen*1e-9 * 1.106292158265238e2  # m^-1/pixel, convert to spatial frequency units
new_pixscale /= 1.106292158265238e2 # focal length of subaru; rad/pixel
new_pixscale *= 206265 # convert to arcsec; as/pixel 
print('new pixelscale ',new_pixscale)

Exx = np.pad(Exx,npix-dim)
Exy = np.pad(Exy,npix-dim)
Eyx = np.pad(Eyx,npix-dim)
Eyy = np.pad(Eyy,npix-dim)

Axx = PropToImage(Exx)
Axy = PropToImage(Exy)
Ayx = PropToImage(Eyx)
Ayy = PropToImage(Eyy)

# Plot to make sure they loaded right
plt.figure(figsize=[10,5])
plt.subplot(241)
plt.imshow(np.abs(Axx),cmap='inferno',norm=LogNorm())
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Axx')

plt.subplot(242)
plt.imshow(np.abs(Axy),cmap='inferno',norm=LogNorm())
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Axy')

plt.subplot(245)
plt.imshow(np.abs(Ayx),cmap='inferno',norm=LogNorm())
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Ayx')

plt.subplot(246)
plt.imshow(np.abs(Ayy),cmap='inferno',norm=LogNorm())
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Ayy')

plt.subplot(243)
plt.imshow(np.angle(Axx),cmap='coolwarm')
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Axx')

plt.subplot(244)
plt.imshow(np.angle(Axy),cmap='coolwarm')
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Axy')

plt.subplot(247)
plt.imshow(np.angle(Ayx),cmap='coolwarm')
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Ayx')

plt.subplot(248)
plt.imshow(np.angle(Ayy),cmap='coolwarm')
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Ayy')
plt.show()

# Print Out the Difference in Beamshift
com_x = np.array(center_of_mass(np.abs(Axx+Axy)))
com_y = np.array(center_of_mass(np.abs(Ayy+Ayx)))

print('Com X-polarization = ',com_x)
print('Com Y-polarization = ',com_y)
print('total beamshift in y and x [as] = ',(com_x-com_y)*new_pixscale)


plt.figure()
plt.imshow(np.log10(np.abs(np.abs(Axx+Axy)**2)),vmax=2,vmin=-10)
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Exx')
plt.show()

plt.figure()
plt.imshow(np.log10(np.abs(np.abs(Ayy+Ayx)**2)),vmax=2,vmin=-10)
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Eyy')
plt.show()

plt.figure()
plt.imshow(np.log10(np.abs(np.abs(Axx+Axy)**2-np.abs(Ayy+Ayx)**2)),vmax=2,vmin=-10)
plt.xlim([npix-cut,npix+cut])
plt.ylim([npix-cut,npix+cut])
plt.colorbar()
plt.title('Image Difference')
plt.show()
