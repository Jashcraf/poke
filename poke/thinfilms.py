from poke.poke_math import np

# Inputs are list of index, distance, and the wavelength
VACUUM_PERMITTIVITY = 8.8541878128e-12  # Farad / M
VACUUM_PERMEABILITY = 1.25663706212e-6  # Henry / M
FREESPACE_IMPEDANCE = 376.730313668  # ohms
FREESPACE_IMPEDANCE_INV = 1 / FREESPACE_IMPEDANCE  # Chipman includes this, Macleod ignores this
ONE_COMPLEX = 1 + 0 * 1j
ZERO_COMPLEX = 0 + 0 * 1j


def compute_thin_films_broadcasted(
    stack, aoi, wavelength, ambient_index=1, substrate_index=1.5, polarization="s"
):
    """compute fresnel coefficients for a multilayer stack using the BYU Optics Book method

    Parameters
    ----------
    stack : list of tuples containing raveled ndarrays, eg. [(n1,d1),(n2,d2),....] 
        The reciple that defines the multilayer stack. where n1.shape,d2.shape = aoi.shape
    aoi : numpy.ndarray
        angle of incidence on the thin film in radians
    wavelength : float
        wavelegnth of the light incident on the thin film stack. Should be in same units as thin film distances.
    ambient_index : float, optional
        index optical system is immersed in, by default 1
    substrate_index : float, optional
        index of substrate thin film is deposited on, by default 1.5
    polarization : str, optional
        polarization state to compute values for, can be 's' or 'p', by default 's'

    Returns
    -------
    rf,tf
        fresnel coefficients for the polarization specified    
    """

    # Do some digesting
    stack = stack[:-1]  # ignore the last element, which contains the substrate index

    # Consider the incident media
    system_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    if len(aoi.shape) > 0:
        system_matrix = np.broadcast_to(system_matrix, [*aoi.shape, *system_matrix.shape])

    # Consider the terminating media
    aor = np.arcsin(ambient_index / substrate_index * np.sin(aoi))
    ncosAOR = substrate_index * np.cos(aor)
    cosAOI = np.cos(aoi)
    sinAOI = np.sin(aoi)
    ncosAOI = ambient_index * cosAOI

    n0 = np.full_like(aoi, ambient_index, dtype=np.complex128)
    nM = np.full_like(aoi, substrate_index, dtype=np.complex128)
    zeros = np.full_like(aoi, 0.0, dtype=np.complex128)
    ones = np.full_like(aoi, 1.0, dtype=np.complex128)

    for layer in stack:

        ni = layer[0]
        di = layer[1]  # has some dimension

        angle_in_film = np.arcsin(ambient_index / ni * sinAOI)

        Beta = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength

        cosB = np.full_like(aoi, np.cos(Beta), dtype=np.complex128)
        sinB = np.full_like(aoi, np.sin(Beta), dtype=np.complex128)
        cosT = np.full_like(aoi, np.cos(angle_in_film), dtype=np.complex128)

        if polarization == "p":
            newfilm = np.array([[cosB, -1j * sinB * cosT / ni], [-1j * ni * sinB / cosT, cosB]])

        elif polarization == "s":
            newfilm = np.array([[cosB, -1j * sinB / (cosT * ni)], [-1j * ni * sinB * cosT, cosB]])
        if newfilm.ndim > 2:
            for i in range(newfilm.ndim - 2):
                newfilm = np.moveaxis(newfilm, -1, 0)

        system_matrix = system_matrix @ newfilm

    # Final matrix
    coeff = 1 / (2 * ncosAOI)
    if system_matrix.ndim > 2:
        coeff = coeff[..., np.newaxis, np.newaxis]

    if polarization == "p":
        front_matrix = np.array([[n0, cosAOI], [n0, -cosAOI]])
        back_matrix = np.array([[np.cos(aor), zeros], [nM, zeros]])

    elif polarization == "s":
        front_matrix = np.array([[ncosAOI, ones], [ncosAOI, -1 * ones]])
        back_matrix = np.array([[ones, zeros], [ncosAOR, zeros]])

    if front_matrix.ndim > 2:
        for i in range(front_matrix.ndim - 2):
            front_matrix = np.moveaxis(front_matrix, -1, 0)

    if back_matrix.ndim > 2:
        for i in range(back_matrix.ndim - 2):
            back_matrix = np.moveaxis(back_matrix, -1, 0)

    characteristic_matrix = coeff * (front_matrix @ system_matrix @ back_matrix)

    ttot = 1 / characteristic_matrix[..., 0, 0]
    rtot = characteristic_matrix[..., 1, 0] / characteristic_matrix[..., 0, 0]

    return rtot, ttot


def compute_thin_films_macleod(
    stack, aoi, wavelength, ambient_index=1, substrate_index=1.5, polarization="s"
):
    """compute fresnel coefficients for a multilayer stack using the Macleod 1969 method

    Parameters
    ----------
    stack : list of tuples containing raveled ndarrays, eg. [(n1,d1),(n2,d2),....] 
        The reciple that defines the multilayer stack. where n1.shape,d2.shape = aoi.shape
    aoi : numpy.ndarray
        angle of incidence on the thin film in radians
    wavelength : float
        wavelegnth of the light incident on the thin film stack. Should be in same units as thin film distances.
    ambient_index : float, optional
        index optical system is immersed in, by default 1
    substrate_index : float, optional
        index of substrate thin film is deposited on, by default 1.5
    polarization : str, optional
        polarization state to compute values for, can be 's' or 'p', by default 's'

    Returns
    -------
    rf,tf
        fresnel coefficients for the polarization specified    
    """

    # Do some digesting
    stack = stack[:-1]  # ignore the last element, which contains the substrate index

    # Consider the incident media
    system_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    if len(aoi.shape) > 0:
        system_matrix = np.broadcast_to(system_matrix, [*aoi.shape, *system_matrix.shape])

    aor = np.arcsin(ambient_index / substrate_index * np.sin(aoi))
    cosAOR = np.cos(aor)
    sinAOI = np.sin(aoi)
    cosAOI = np.cos(aoi)
    ones = np.ones_like(aoi)

    # compute the substrate admittance
    if polarization == "s":
        n_substrate = substrate_index * cosAOR
        eta_medium = ambient_index * cosAOI
    elif polarization == "p":
        n_substrate = substrate_index / cosAOR
        eta_medium = ambient_index / cosAOI
    else:
        print("polarization not recognized, defaulting to s")
        n_substrate = substrate_index * cosAOR

    for layer in stack:

        ni = layer[0]
        di = layer[1]  # has some dimension

        angle_in_film = np.arcsin(ambient_index / ni * sinAOI)

        # phase thickness
        Beta = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength
        cosB = np.cos(Beta)
        isinB = 1j * np.sin(Beta)

        # film admittance
        if polarization == "p":
            eta_film = ni / np.cos(angle_in_film)
        else:
            eta_film = ni * np.cos(angle_in_film)

        # assemble the characteristic matrix
        newfilm = np.array([[cosB, isinB / eta_film], [isinB * eta_film, cosB]])

        if newfilm.ndim > 2:
            for i in range(newfilm.ndim - 2):
                newfilm = np.moveaxis(newfilm, -1, 0)

        # apply the matrix
        system_matrix = system_matrix @ newfilm

    # layer finished
    substrate_vec = np.array([ones, n_substrate])
    substrate_vec = np.swapaxes(substrate_vec, -1, 0)
    substrate_vec = substrate_vec[..., np.newaxis]
    BC = system_matrix @ substrate_vec
    Y = BC[..., 1, 0] / BC[..., 0, 0]

    rtot = (eta_medium - Y) / (eta_medium + Y)
    return rtot
