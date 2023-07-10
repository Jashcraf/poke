"""interfaces with POPPY, HCIPy"""
import numpy as np

def jones_pupil_to_hcipy_wavefront(jones_pupil,pupil_grid,input_stokes_vector=[1,0,0,0],shape=None):
    """converts a poke jones pupil to an HCIPy partially polarized wavefront,
    only works on square jones pupils

    Parameters
    ----------
    jones_pupil : numpy.ndarray
        poke jones pupil, typically Rayfront.jones_pupil
    pupil_grid : hcipy.PupilGrid
        Pupil grid from hcipy that the jones pupil is defined on
    input_stokes_vector : list, optional
        stokes vector of the wavefront, by default [1,0,0,0]
    shape : int, optional
        dimension on the side of the square jones pupil, by default None

    Returns
    -------
    wavefront
        hcipy.Wavefront containing the Jones pupil data
    """
    # First test the import
    try:
        
        import hcipy

    except Exception as e:

        print(f'Error trying to import HCIPy \n {e}')

    # Next test the ability to reshape the jones pupil
    # TODO: Add option to fit data to Zernike polynomials
    # try:

    if functional_decomposition:
        
        regularly_space_jones(rayfront,nmodes,npix,which=-1)

    else:

        if shape is None:

            size = jones_pupil[-1][...,0,0].shape[0]
            shape = np.sqrt(size)
            shape = int(np.sqrt(size))

        jones_reshaped = jones_pupil[-1][...,:2,:2]
        field = hcipy.Field([[jones_reshaped[...,0,0],jones_reshaped[...,0,1]],
                            [jones_reshaped[...,1,0],jones_reshaped[...,1,1]]],pupil_grid)
        
    wavefront = hcipy.Wavefront(field,input_stokes_vector=input_stokes_vector)

    return wavefront

def jones_pupil_to_poppy_wavefronts(jones_pupil,wavelength=1e-6,shape=None):
    """converts a Poke jones pupil to a POPPY wavefront list

    Parameters
    ----------
    jones_pupil : numpy.ndarray
        poke jones pupil, typically Rayfront.jones_pupil
    wavelength : float, optional
        wavelength in meters, by default 1e-6
    shape : int, optional
        dimension on the side of the square jones pupil, by default None

    Returns
    -------
    wflist
        list of poppy.Wavefronts
    """

    try:
        import poppy
        import astropy.units as u

    except Exception as e:
        print(f'Error trying to import POPPY and/or astropy \n {e}')

    if shape is None:
        size = jones_pupil[-1][...,0,0].shape[0]
        shape = int(np.sqrt(size))

    jones_reshaped = jones_pupil[-1][...,:2,:2].reshape([shape,shape,2,2])
    jones_pupils = [jones_reshaped[...,0,0],jones_reshaped[...,0,1],jones_reshaped[...,1,0],jones_reshaped[...,1,1]]
    wflist = []
    for jones in jones_pupils:
        wf = poppy.Wavefront(wavelength=wavelength*u.m,npix=shape,diam=1*u.m,oversample=1)
        wf.wavefront = jones
        wflist.append(wf)

    return wflist

def rayfront_to_hcipy_wavefront(rayfront,npix,pupil_grid,nmodes=11,input_stokes_vector=[1,0,0,0],which=-1):

    # First test the import
    try:
        
        import hcipy

    except Exception as e:

        print(f'Error trying to import HCIPy \n {e}')

    jones_pupil = regularly_space_jones(rayfront,nmodes,npix,which=which)
    field = hcipy.Field([[jones_pupil[...,0,0].ravel(),jones_pupil[...,0,1].ravel()],
                         [jones_pupil[...,1,0].ravel(),jones_pupil[...,1,1].ravel()]],pupil_grid)
    wavefront = hcipy.Wavefront(field,input_stokes_vector=input_stokes_vector)
    return wavefront

def zernike(rho, phi, J):
    """contributed by Emory Jenkins"""
    N=int(np.ceil(np.sqrt(2*J + 0.25)-0.5)) # N = number of rows on the zernike pyramid
    values = np.zeros([rho.size, J+1])
    j=0 # ANSI index of zernike
    for n in range(0,N):
        m=-n
        while m<=n:
            R = 0
            for k in range(0,1+(n-abs(m))//2):
                c_k = ((-1)**k * np.math.factorial(n-k))/(np.math.factorial(k) * np.math.factorial((n+abs(m))//2 - k) * np.math.factorial((n-abs(m))//2 - k))
                R += c_k * rho**(n-2*k)
            if m>0:
                Z = R*np.cos(m*phi)
            elif m<0:
                Z = R*np.sin(abs(m)*phi)
            else:
                Z = R
            values[:,j] = Z
            j=j+1
            if j>J:
                break
            m=m+2
    return values

def regularly_space_jones(rayfront,nmodes,npix,which=-1):

    jones_pupil = rayfront.jones_pupil[which]

    # TODO: This breaks for decentered pupils, need to implement an offset
    x,y = rayfront.xData[0,0],rayfront.yData[0,0]
    x = x/np.max(x)
    y = y/np.max(y)
    r,t = np.sqrt(x**2 + y**2),np.arctan2(y,x)
    irregularly_spaced_basis = zernike(r,t,nmodes)

    cxx = np.linalg.lstsq(irregularly_spaced_basis,jones_pupil[...,0,0],rcond=None)
    cxy = np.linalg.lstsq(irregularly_spaced_basis,jones_pupil[...,0,1],rcond=None)
    cyx = np.linalg.lstsq(irregularly_spaced_basis,jones_pupil[...,1,0],rcond=None)
    cyy = np.linalg.lstsq(irregularly_spaced_basis,jones_pupil[...,1,1],rcond=None)

    x = np.linspace(-1,1,npix)
    x,y = np.meshgrid(x,x)
    r,t = np.sqrt(x**2 + y**2),np.arctan2(y,x)
    regularly_spaced_basis = zernike(r.ravel(),t.ravel(),nmodes)

    return_jones = np.empty([npix,npix,2,2],dtype=np.complex128)
    return_jones[...,0,0] = np.sum(regularly_spaced_basis*cxx[0],axis=-1).reshape([npix,npix])
    return_jones[...,0,1] = np.sum(regularly_spaced_basis*cxy[0],axis=-1).reshape([npix,npix])
    return_jones[...,1,0] = np.sum(regularly_spaced_basis*cyx[0],axis=-1).reshape([npix,npix])
    return_jones[...,1,1] = np.sum(regularly_spaced_basis*cyy[0],axis=-1).reshape([npix,npix])

    return return_jones

    

    