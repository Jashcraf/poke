# dependencies
from poke.poke_math import (
    np,
    mat_inv_3x3,
    vector_norm,
    vector_angle,
    rotation_3d,
    broadcast_kron,
    vectorAngle,
    rotation3D,
)
import poke.thinfilms as tf
import poke.poke_math as math
import matplotlib.pyplot as plt
from .conf import config

# def plot3x3(raybundle,op=np.abs):
#     """plots a 3x3 matrix"""

#     x = raybundle.xData[0,0]
#     y = raybundle.yData[0,0]

#     fig,ax = plt.subplots(nrows=3,ncols=3)
#     for row in range(3):
#         for column in range(3):

#             ax[row,column].scatter(x,y,c=op(raybundle.P_total[0][...,row,column]))
#     plt.show()


def fresnel_coefficients(aoi, n1, n2, mode="reflect"):
    """Computes Fresnel Coefficients for a single surface interaction

    Parameters
    ----------
    aoi : float or array of floats
        angle of incidence in radians on the interface
    n1 : float 
        complex refractive index of the incident media
    n2 : float
        complex refractive index of the exitant media

    Returns
    -------
    fs, fp: complex floats
        the Fresnel s- and p- coefficients of the surface interaction
    """

    if (mode != "reflect") and (mode != "transmit"):
        print("not a valid mode, please use reflect, transmit, or both. Defaulting to reflect")
        mode = "reflect"

    # ratio of refractive indices
    n = n2 / n1

    if mode == "reflect":

        fs = (np.cos(aoi) - np.sqrt(n ** 2 - np.sin(aoi) ** 2)) / (
            np.cos(aoi) + np.sqrt(n ** 2 - np.sin(aoi) ** 2))
        
        fp = (n ** 2 * np.cos(aoi) - np.sqrt(n ** 2 - np.sin(aoi) ** 2)) / (
            n ** 2 * np.cos(aoi) + np.sqrt(n ** 2 - np.sin(aoi) ** 2))

    elif mode == "transmit":

        fs = (2 * np.cos(aoi)) / (np.cos(aoi) + np.sqrt(n ** 2 - np.sin(aoi) ** 2))
        fp = (2 * n * np.cos(aoi)) / (n ** 2 * np.cos(aoi) + np.sqrt(n ** 2 - np.sin(aoi) ** 2))

    return fs, fp


def _fresnel_coefficients(aoi, n1, n2, mode="reflect"):
    """Computes Fresnel Coefficients for a single surface interaction

    Parameters
    ----------
    aoi : float or array of floats
        angle of incidence in radians on the interface
    aor : float or array of floats
        angle of refraction in radians on the interface
    n1 : float 
        complex refractive index of the incident media
    n2 : float
        complex refractive index of the exitant media

    Returns
    -------
    fs, fp: complex floats
        the Fresnel s- and p- coefficients of the surface interaction
    """
    cosaoi = np.cos(aoi)
    aor = np.arcsin(n1 * np.sin(aoi) / n2)
    cosaor = np.cos(aor)

    if (mode != "reflect") and (mode != "transmit"):
        print("not a valid mode, please use reflect, transmit, or both. Defaulting to reflect")
        mode = "reflect"

    elif mode == "reflect":
        fs = (n1 * cosaoi - n2 * cosaor) / (n1 * cosaoi + n2 * cosaor)
        fp = (n2 * cosaoi - n1 * cosaor) / (n2 * cosaoi + n1 * cosaor) 

    elif mode == "transmit":
        fs = (2 * n1 * cosaoi) / (n1 * cosaoi + n2 * cosaor)
        fp = (2 * n1 * cosaoi) / (n2 * cosaoi + n1 * cosaor)

    return fs, fp

def orthogonal_transofrmation_matrices(kin, kout, normal):
    """compute the orthogonal transformation matrices that rotate into and out of the local coordinates
    of a surface

    Parameters
    ----------
    kin : numpy.ndarray
        array containing the incident ray vectors
    kout : numpy.ndarray
        array containing the exitant ray vectors
    normal : numpy.ndarray
        array containing the surface normal vectors

    Returns
    -------
    Oinv,Oout : numpy.ndarray
        orthogonal transformation matrices
    """

    # ensure wave vectors are normalized
    kin = kin / vector_norm(kin)[..., np.newaxis]
    kout = kout / vector_norm(kout)[..., np.newaxis]

    # get s-basis vector
    sin = np.cross(kin, normal)
    sin = sin / vector_norm(sin)[..., np.newaxis]

    # get p-basis vector
    pin = np.cross(kin, sin)
    pin = pin / vector_norm(pin)[..., np.newaxis]

    # Assemble Oinv
    Oinv = np.array([sin, pin, kin])
    Oinv = np.moveaxis(Oinv, -1, 0)
    if Oinv.ndim > 2:
        for i in range(Oinv.ndim - 2):
            Oinv = np.moveaxis(Oinv, -1, 0)
    Oinv = np.swapaxes(Oinv, -1, -2)  # take the transpose/inverse

    # outgoing basis vectors
    sout = sin
    pout = np.cross(kout, sout)
    pout = pout / vector_norm(pout)[..., np.newaxis]
    Oout = np.array([sout, pout, kout])
    Oout = np.moveaxis(Oout, -1, 0)
    if Oout.ndim > 2:
        for i in range(Oout.ndim - 2):
            Oout = np.moveaxis(Oout, -1, 0)
    # Oout = np.moveaxis(Oout,0,-1)

    return Oinv, Oout


def _prt_matrix(kin, kout, normal, aoi, surfdict, wavelength, ambient_index):
    """prt matrix for a single surface

    Parameters
    ----------
    aoi : numpy.ndarray
        array describing the ray angles of incidence on a surface
    kin : numpy.ndarray
        array containing the incident ray vectors
    kout : numpy.ndarray
        array containing the exitant ray vectors
    norm : numpy.ndarray
        array containing the surface normal vectors
    surfdict : dict
        dictionary describing the surface interaction
    wavelength : float
        wavelength of light the computation is done at
    ambient_index : float
        refractive index of material in the exiting space

    Returns
    -------
    Pmat,J,Qmat
        PRT, Jones, and parallel transport matrices for a given surface
    """

    normal = -normal
    offdiagbool = False

    # A surface decision tree - TODO: it is worth trying to make this more robust
    if type(surfdict["coating"]) == list:

        # prysm likes films in degress, wavelength in microns, thickness in microns
        if config.refractive_index_sign == "positive":
            rs, ts = tf.compute_thin_films_broadcasted(
                surfdict["coating"],
                aoi,
                wavelength,
                substrate_index=surfdict["coating"][-1],
                polarization="s",
            )
            rp, tp = tf.compute_thin_films_broadcasted(
                surfdict["coating"],
                aoi,
                wavelength,
                substrate_index=surfdict["coating"][-1],
                polarization="p",
            )

        elif config.refractive_index_sign == "negative":
            rs, ts = tf.compute_thin_films_macleod(
                surfdict["coating"],
                aoi,
                wavelength,
                substrate_index=surfdict["coating"][-1],
                polarization="s",
            )
            rp, tp = tf.compute_thin_films_macleod(
                surfdict["coating"],
                aoi,
                wavelength,
                substrate_index=surfdict["coating"][-1],
                polarization="p",
            )
        
        else:
            raise ValueError('Set the refractive index sign in poke.conf to "positive" or "negative"')

        if surfdict["mode"] == "reflect":
            fss = rs
            fpp = rp * np.exp(-1j * np.pi)  # The Thin Film Correction

        if surfdict["mode"] == "transmit":
            fss = ts
            fpp = tp

    elif (type(surfdict["coating"]) == np.ndarray):  # assumes the film is defined with first index as fs,fp

        fss = surfdict["coating"][0, 0]
        fsp = surfdict["coating"][0, 1]
        fps = surfdict["coating"][1, 0]
        fpp = surfdict["coating"][1, 1]
        offdiagbool = True

    elif callable(surfdict["coating"]):  # check if a function
        fss, fps = surfdict["coating"](aoi)

    else:

        fss, fpp = fresnel_coefficients(aoi, ambient_index, surfdict["coating"], mode=surfdict["mode"])
        if np.imag(surfdict["coating"]) < 0:  # TODO: This is a correction for the n - ik configuration, need to investigate if physical
            fss *= np.exp(-1j * np.pi)
            fpp *= np.exp(1j * np.pi)

    Oinv, Oout = orthogonal_transofrmation_matrices(kin, kout, normal)

    # Compute the Jones matrix and parallel transport matrix
    zeros = np.zeros(fss.shape)
    ones = np.ones(fss.shape)
    if offdiagbool:
        J = np.asarray([[fss, fsp, zeros], [fps, fpp, zeros], [zeros, zeros, ones]])
    else:
        J = np.asarray([[fss, zeros, zeros], [zeros, fpp, zeros], [zeros, zeros, ones]])
    B = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # dimensions need to be appropriate
    if J.ndim > 2:
        for _ in range(J.ndim - 2):
            J = np.moveaxis(J, -1, 0)

    # compute PRT matrix and orthogonal transformation
    Pmat = Oout @ J @ Oinv
    Qmat = Oout @ B @ Oinv  # test if this broadcasts

    return Pmat, J, Qmat


def prt_matrix(kin, kout, normal, aoi, surfdict, wavelength, ambient_index):
    """prt matrix for a single surface

    Parameters
    ----------
    aoi : numpy.ndarray
        array describing the ray angles of incidence on a surface
    kin : numpy.ndarray
        array containing the incident ray vectors
    kout : numpy.ndarray
        array containing the exitant ray vectors
    norm : numpy.ndarray
        array containing the surface normal vectors
    surfdict : dict
        dictionary describing the surface interaction
    wavelength : float
        wavelength of light the computation is done at
    ambient_index : float
        refractive index of material in the exiting space

    Returns
    -------
    Pmat,J,Qmat
        PRT, Jones, and parallel transport matrices for a given surface
    """

    normal = -normal
    offdiagbool = False

    if surfdict["mode"] == "reflect":

        # List of tuples means multilayer stack
        if type(surfdict["coating"]) == list:

            if config.refractive_index_sign == "positive":
                fss, ts = tf.compute_thin_films_broadcasted(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    substrate_index=surfdict["coating"][-1],
                    polarization="s")
                
                fpp, tp = tf.compute_thin_films_broadcasted(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    substrate_index=surfdict["coating"][-1],
                    polarization="p")
                
                # pi phase shift on reflection
                fpp = fpp * np.exp(-1j * np.pi)

            elif config.refractive_index_sign == "negative":
                fss, ts = tf.compute_thin_films_macleod(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    substrate_index=surfdict["coating"][-1],
                    polarization="s")
                
                fpp, tp = tf.compute_thin_films_macleod(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    substrate_index=surfdict["coating"][-1],
                    polarization="p")
                
                # pi phase shift on reflection
                fpp = fpp * np.exp(-1j * np.pi)
                
            else:
                raise ValueError('Set the refractive index sign in poke.conf to "positive" or "negative"')
        
        # User-specified jones matrix
        elif type(surfdict["coating"]) == np.ndarray:

            fss = surfdict["coating"][0, 0]
            fsp = surfdict["coating"][0, 1]
            fps = surfdict["coating"][1, 0]
            fpp = surfdict["coating"][1, 1]
            offdiagbool = True

        # check if coating is a function of angle of incidence
        elif callable(surfdict["coating"]):
            fss, fps = surfdict["coating"](aoi)

        # treat as a float
        else:
            fss, fpp = fresnel_coefficients(aoi, ambient_index, surfdict["coating"], mode=surfdict["mode"])
            if np.imag(surfdict["coating"]) < 0:  # TODO: This is a correction for the n - ik configuration, need to investigate if physical
                fss *= np.exp(-1j * np.pi)
                fpp *= np.exp(1j * np.pi)


    elif surfdict["mode"] == "transmit":

        # List of tuples means multilayer stack
        if type(surfdict["coating"]) == list:

            if config.refractive_index_sign == "positive":
                rs, fss = tf.compute_thin_films_broadcasted(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    ambient_index=surfdict["coating"][0],
                    substrate_index=surfdict["coating"][-1],
                    polarization="s")
                
                rp, fpp = tf.compute_thin_films_broadcasted(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    ambient_index=surfdict["coating"][0],
                    substrate_index=surfdict["coating"][-1],
                    polarization="p")

            elif config.refractive_index_sign == "negative":
                rs, fss = tf.compute_thin_films_macleod(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    ambient_index=surfdict["coating"][0],
                    substrate_index=surfdict["coating"][-1],
                    polarization="s")
                
                rp, fpp = tf.compute_thin_films_macleod(
                    surfdict["coating"],
                    aoi,
                    wavelength,
                    ambient_index=surfdict["coating"][0],
                    substrate_index=surfdict["coating"][-1],
                    polarization="p")
            
            else:
                raise ValueError('Set the refractive index sign in poke.conf to "positive" or "negative"')
        
        # User-specified jones matrix
        elif type(surfdict["coating"]) == np.ndarray:

            fss = surfdict["coating"][0, 0]
            fsp = surfdict["coating"][0, 1]
            fps = surfdict["coating"][1, 0]
            fpp = surfdict["coating"][1, 1]
            offdiagbool = True

        # check if coating is a function of angle of incidence
        elif callable(surfdict["coating"]):
            fss, fpp = surfdict["coating"](aoi)

        # treat as a float
        else:
            fss, fpp = fresnel_coefficients(aoi, surfdict["coating"][0], surfdict["coating"][1], mode=surfdict["mode"])

    else:
        raise ValueError(f"surfdict mode {surfdict['mode']} on surface {surfdict['surf']} not recognized, please use 'reflect' or 'transmit'")


    Oinv, Oout = orthogonal_transofrmation_matrices(kin, kout, normal)

    # Compute the Jones matrix and parallel transport matrix
    zeros = np.zeros(fss.shape)
    ones = np.ones(fss.shape)

    # Check if off-diagonal was populated
    if offdiagbool:
        J = np.asarray([[fss, fsp, zeros],
                        [fps, fpp, zeros],
                        [zeros, zeros, ones]])
    
    else:
        J = np.asarray([[fss, zeros, zeros],
                        [zeros, fpp, zeros],
                        [zeros, zeros, ones]])

    # Matrix for parallel transport
    B = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    # dimensions need to be appropriate
    if J.ndim > 2:
        for _ in range(J.ndim - 2):
            J = np.moveaxis(J, -1, 0)

    # compute PRT matrix and orthogonal transformation
    Pmat = Oout @ J @ Oinv
    Qmat = Oout @ B @ Oinv

    return Pmat, J, Qmat


def system_prt_matrices(aoi, kin, kout, norm, surfaces, wavelength, ambient_index):
    """computes the PRT matrices for each surface in the optical system

    Parameters
    ----------
    aoi : numpy.ndarray
        array describing the ray angles of incidence on a surface
    kin : numpy.ndarray
        array containing the incident ray vectors
    kout : numpy.ndarray
        array containing the exitant ray vectors
    norm : numpy.ndarray
        array containing the surface normal vectors
    surfaces : list
        list of dictionaries describing the surface interaction
    wavelength : float
        wavelength of light the computation is done at
    ambient_index : float
        refractive index that the optical system is immersed in

    Returns
    -------
    P,J,Q
        lists of the PRT matrices, Jones matrices, and parallel transport matrices
    """

    P = []
    J = []
    Q = []

    for i, surfdict in enumerate(surfaces):

        kisurf = np.moveaxis(kin[i], -1, 0)
        kosurf = np.moveaxis(kout[i], -1, 0)
        normsurf = np.moveaxis(norm[i], -1, 0)
        aoisurf = np.moveaxis(aoi[i], -1, 0)

        Pmat, Jmat, Qmat = prt_matrix(kisurf, kosurf, normsurf, aoisurf, surfdict, wavelength, ambient_index)

        P.append(Pmat)
        J.append(Jmat)
        Q.append(Qmat)

    return P, J, Q


def total_prt_matrix(P, Q):
    """computes the total PRT matrix for the optical system

    Parameters
    ----------
    P : list
        prt matrices computed per surface
    Q : list
        unpolarized prt matrices computed per surface. Largely for berry phase calculations

    Returns
    -------
    numpy.ndarrays
        the total PRT and Parallel transport matrices
    """

    for i, (p, q) in enumerate(zip(P, Q)):

        if i == 0:
            Ptot = p
            Qtot = q

        else:
            Ptot = p @ Ptot
            Qtot = q @ Qtot

    return Ptot, Qtot


def global_to_local_coordinates(P, kin, k, a, xin, exit_x, Q=None, coordinates="double"):
    """Use the double pole basis to compute the local coordinate system of the Jones pupil.
    Vectorized to perform on arrays of arbitrary shape, assuming the PRT matrix is in the last
    two dimensions.
    Chipman, Lam, Young, from Ch 11 : The Jones Pupil

    Parameters
    ----------
    Pmat : ndarray
        Pmat is the polarization ray tracing matrix
    kin : ndarray
        incident direction cosine vector at the entrance pupil. Generally an ND array where
        the vector is in the last dimension.
    kout : ndarray
        exiting direction cosine vector at the exit pupil. Generally an ND array where
        the vector is in the last dimension.
    a : ndarray
        vector in global coordinates describing the antipole direction
    xin : ndarray
        vector in global coordinates describing the input local x direction
    exit_x : ndarray
        vector in global coordinates describing the direction that should be the 
        "local x" direction
    Q : Parallel Transport matrix
        the non-polarizing PRT matrix, used to account for geometric transformations
    coordinates : string
        type of local coordinate transformation to use. Options are "double" for the double-pole
        coordinate system, and "dipole" for the dipole coordinate system.

    Returns
    -------
    J : ndarray
        shape 3 x 3 ndarray containing the Jones pupil of the optical system. The elements
        Jtot[0,2], Jtot[1,2], Jtot[2,0], Jtot[2,1] should be zero.
        Jtot[-1,-1] should be 1
    """

    # Double Pole Coordinate System, requires a rotation about an axis
    # Wikipedia article seems to disagree with CLY Example 11.4
    # Default entrance pupil in Zemax. Note that this assumes the stop is at the first surface
    
    kin = np.moveaxis(kin, -1, 0)
    k = np.moveaxis(k, -1, 0)
    xin = xin / vector_norm(xin)[..., np.newaxis]

    if coordinates == "double":

        # Do the entrance coordinates
        kin = kin / vector_norm(kin)[..., np.newaxis]
        rin = np.cross(kin, a)
        rin = rin / vector_norm(rin)[..., np.newaxis]
        thin = -vector_angle(kin, a)
        Rin = rotation_3d(thin, rin)
        yin = np.cross(a, xin)
        yin = yin / vector_norm(yin)[..., np.newaxis]
        xin = Rin @ xin
        yin = Rin @ yin

        k = k / vector_norm(k)[..., np.newaxis]
        r = np.cross(k, a)
        r = r / vector_norm(r)[..., np.newaxis]
        th = -vector_angle(k, a)
        R = rotation_3d(th, r)

        # Local basis vectors
        xout = exit_x
        yout = np.cross(a, xout)
        yout /= vector_norm(yout)

        x = R @ xout
        y = R @ yout

    elif coordinates == "dipole":
        
        xin = np.broadcast_to(xin, kin.shape)
        yin = np.cross(kin, xin)
        yin = yin / vector_norm(yin)[..., np.newaxis]
        yin = np.broadcast_to(yin, kin.shape)

        k = k / vector_norm(k)[..., np.newaxis]
        x = np.cross(a, k)
        x = x / vector_norm(x)[..., np.newaxis]

        y = np.cross(k, x)
        y = y / vector_norm(y)[..., np.newaxis]

    else:
        raise ValueError(f"Coordinate '{coordinates}' not recognized, please use 'double' or 'dipole'")
    
    # set up orthogonal transformations
    O_e = np.array(
        [
            [xin[..., 0], yin[..., 0], kin[..., 0]],
            [xin[..., 1], yin[..., 1], kin[..., 1]],
            [xin[..., 2], yin[..., 2], kin[..., 2]],
        ]
    )
    O_e = np.moveaxis(O_e, -1, 0)

    O_x = np.array(
        [
            [x[..., 0], y[..., 0], k[..., 0]],
            [x[..., 1], y[..., 1], k[..., 1]],
            [x[..., 2], y[..., 2], k[..., 2]],
        ]
    )

    O_x = np.moveaxis(O_x, -1, 0)

    # apply proper retardance correction
    if type(Q) == np.ndarray:
        P = mat_inv_3x3(Q) @ P

    J = mat_inv_3x3(O_x) @ P @ O_e

    return J


def jones_to_mueller(Jones):
    """Converts a Jones matrix to a Mueller matrix

    Parameters
    ----------
    Jones : 2x2 ndarray
        Jones matrix to convert to a mueller matrix

    Returns
    -------
    M
        Mueller matrix from Jones matrix
    """

    U = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])

    U *= np.sqrt(1 / 2)

    M = np.real(U @ (np.kron(np.conj(Jones), Jones)) @ np.linalg.inv(U))

    return M


def jones_to_mueller_broadcast(jones):

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,1j,-1j,0]])

    U *= np.sqrt(1/2)
    Uinv = np.linalg.inv(U)
    inner = broadcast_kron(jones.conj(), jones)

    M = np.real(U @ inner @ Uinv)
    return M


def mueller_to_jones(M):
    """Converts Mueller matrix to a relative Jones matrix. Phase aberration is relative to the Pxx component.

    Returns
    -------
    J : 2x2 ndarray
        Jones matrix from Mueller matrix calculation
    """

    "CLY Eq. 6.112"
    "Untested"

    print("warning : This operation looses global phase")

    pxx = np.sqrt((M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]) / 2)
    pxy = np.sqrt((M[0, 0] - M[0, 1] + M[1, 0] - M[1, 1]) / 2)
    pyx = np.sqrt((M[0, 0] + M[0, 1] - M[1, 0] - M[1, 1]) / 2)
    pyy = np.sqrt((M[0, 0] - M[0, 1] - M[1, 0] + M[1, 1]) / 2)

    txx = 0  # This phase is not determined
    txy = -np.arctan((M[0, 3] + M[1, 3]) / (M[0, 2] + M[1, 2]))
    tyx = np.arctan((M[3, 0] + M[3, 1]) / (M[2, 0] + M[2, 1]))
    tyy = np.arctan((M[3, 2] - M[2, 3]) / (M[2, 2] + M[3, 3]))

    J = np.array(
        [
            [pxx * np.exp(-1j * txx), pxy * np.exp(-1j * txy)],
            [pyx * np.exp(-1j * tyx), pyy * np.exp(-1j * tyy)],
        ]
    )

    return J