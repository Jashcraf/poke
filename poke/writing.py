import numpy as np
from astropy.io import fits

def WriteMatrixToFITS(matrix,filename):

    # Assumes an N x N x Npts matrix that you want to interpolate and write onto a grid. Npts should have an integer root

    # grab matrix dimension
    dim = matrix.shape[0]

    # Total number of points
    Npts = matrix.shape[-1]
    
    # number of points on one side of the square
    npts = int(np.sqrt(Npts))

    # simplifying dimensions would be better - c'est la vie
    ims_abs = np.empty([npts,npts,dim-1,dim-1],dtype='float64')
    ims_phs = np.empty([npts,npts,dim-1,dim-1],dtype='float64')

    # minus one grabs the corner
    for i in range(matrix.shape[0]-1):
        for j in range(matrix.shape[1]-1):

            scattered_matrix = matrix[i,j,:]
            reshaped_matrix = np.reshape(scattered_matrix,[npts,npts])

            real_part = np.abs(reshaped_matrix)
            imag_part = np.angle(reshaped_matrix)

            ims_abs[:,:,i,j] = real_part
            ims_phs[:,:,i,j] = imag_part

    # Now write to fits file
    hdul_abs = fits.HDUList([fits.PrimaryHDU(ims_abs)])
    # hdul_abs.Header.set('AXIS0','npix')
    # hdul_abs.Header.set('AXIS1','npix')
    # hdul_abs.Header.set('AXIS2','Y index')
    # hdul_abs.Header.set('AXIS3','X index')
    hdul_phs = fits.HDUList([fits.PrimaryHDU(ims_phs)])
    # hdul_phs.header.set('AXIS0','npix')
    # hdul_phs.header.set('AXIS1','npix')
    # hdul_phs.header.set('AXIS2','Y index')
    # hdul_phs.header.set('AXIS3','X index')
    
    hdul_abs.writeto(filename+'_amplitude.fits',overwrite=True)
    hdul_phs.writeto(filename+'_phase.fits',overwrite=True)