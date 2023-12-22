"""interfaces with POPPY, HCIPy"""
from poke.poke_math import np

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
    """convert rayfront to an hcipy wavefront using zernike decomposition

    Parameters
    ----------
    rayfront : poke.Rayfront
        rayfront which contains the jones pupils to perform the decomposition on
    npix : int
        number of pixels along the side of the jones pupils this function returns
    pupil_grid : hcipy pupil grid
        HCIPy pupil grid to define the Wavefront on
    nmodes : int, optional
        number of Zernike modes to use in the decomposition, by default 11
    input_stokes_vector : list, optional
        stokes vector that defines the polarization of the wavefront, by default [1,0,0,0]
    which : int, optional
        which jones pupil in the Rayfront.jones_pupil list to use, by default -1

    Returns
    -------
    hcipy.Wavefront
        hcipy partially polarized Wavefront defined with the Jones pupil
    """

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
    """Generates an array containing Zernike polynomials
    contributed by Emory Jenkins with edits made by Jaren Ashcraft

    Parameters
    ----------
    rho : numpy.ndarray
        radial coordinate
    phi : numpy.ndarray
        azimuthal coordinate
    J : int
        maximum number of modes to use, Noll indexed

    Returns
    -------
    values : numpy.ndarray
        array containing the Zernike modes
    """
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

def regularly_space_jones(rayfront,nmodes,npix,which=-1,return_residuals=False):
    """converts a jones pupil from a rayfront to a regularly-spaced array with zernike decomposition

    Parameters
    ----------
    rayfront : poke.Rayfront
        Rayfront that holds the jones pupil
    nmodes : int
        number of modes to use in the decomposition
    npix : int
        number of samples along the side of the output array
    which : int, optional
        which jones pupil in the rf.jones_pupil list to use, by default -1
    return_residuals : bool, optional
        Whether to return the full np.linalg.lstsq residuals, by default False

    Returns
    -------
    numpy.ndarray
        npix x npix x 2 x 2 array containing the jones pupil data
    """

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

    if return_residuals:
        return return_jones, [cxx,cxy,cyx,cyy]
    else:
        return return_jones

    

    