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
    ims = np.empty([npts,npts,dim,dim,2],dtype='float64')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            scattered_matrix = matrix[i,j,:]
            reshaped_matrix = np.reshape(scattered_matrix,[npts,npts])

            real_part = np.real(reshaped_matrix)
            imag_part = np.imag(reshaped_matrix)

            ims[:,:,i,j,0] = real_part
            ims[:,:,i,j,1] = imag_part

    # Now write to fits file
    hdu = fits.PrimaryHDU(ims)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename,overwrite=True)