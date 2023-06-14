"""interfaces with POPPY, HCIPy"""
import numpy as np

def jones_pupil_to_hcipy_wavefront(jones_pupil,pupil_grid,input_stokes_vector=[1,0,0,0],shape=None):
    """converts a poke jones pupil to an HCIPy partially polarized wavefront

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

    if shape is None:
        size = jones_pupil[-1][...,0,0].shape[0]
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
    

    