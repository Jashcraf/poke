from poke.poke_math import np

# Inputs are list of index, distance, and the wavelength
VACUUM_PERMITTIVITY = 8.8541878128e-12 # Farad / M
VACUUM_PERMEABILITY =  1.25663706212e-6 # Henry / M
FREESPACE_IMPEDANCE = 376.730313668 # ohms
FREESPACE_IMPEDANCE_INV = 1/FREESPACE_IMPEDANCE # Chipman includes this, Macleod ignores this
ONE_COMPLEX = 1 + 0*1j
ZERO_COMPLEX = 0 + 0*1j

def compute_thin_films_macleod(stack, aoi, wavelength, ambient_index=1, substrate_index=1.5):
    """
    Parameters
    ----------

    stack : list of tuples
        list composed of elements containing the index (n) and thickness (t) in meters, ordered like
        stack = [(n0,t0),(n1,t1)...,(nN,tN)]. nN and tN are of the same shape, but can be any shape.

    aoi : float
        angle of incidence in radians on the thin film stack

    wavelength: float
        wavelength to comput the reflection coefficients for in meters

    """

    # Consider the incident media
    system_matrix_s = np.array([[1, 0], [0, 1]], dtype='complex128')
    system_matrix_p = np.copy(system_matrix_s)

    eta_ambient_s = ambient_index*np.cos(aoi)
    eta_ambient_p = ambient_index/np.cos(aoi)

    # Consider the terminating media
    aor = np.lib.scimath.arcsin(ambient_index/substrate_index*np.sin(aoi))
    eta_substrate_s = substrate_index*np.cos(aor)
    eta_substrate_p = substrate_index/np.cos(aor)

    for layer in stack:

        ni = layer[0]
        di = layer[1]

        angle_in_film = np.lib.scimath.arcsin(ambient_index/ni*np.sin(aoi))

        phase_thickness = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength
        eta_film_s = ni*np.cos(angle_in_film)
        eta_film_p = ni/np.cos(angle_in_film)

        newfilm_s = np.array([[np.cos(phase_thickness), 1j*np.sin(phase_thickness)/eta_film_s],
                              [1j*eta_film_s*np.sin(phase_thickness), np.cos(phase_thickness)]])

        newfilm_p = np.array([[np.cos(phase_thickness), 1j*np.sin(phase_thickness)/eta_film_p],
                              [1j*eta_film_p*np.sin(phase_thickness), np.cos(phase_thickness)]])
        
        if newfilm_s.ndim > 2:
            newfilm_p = np.moveaxis(newfilm_p, -1, 0)
        if newfilm_p.ndim > 2:
            newfilm_p = np.moveaxis(newfilm_p,-1,0)

        system_matrix_s = system_matrix_s @ newfilm_s
        system_matrix_p = system_matrix_p @ newfilm_p

    # Computes the s-vector
    s_vector_substrate = np.array([np.full_like(eta_substrate_s, 1), eta_substrate_s])
    if s_vector_substrate.ndim >2:
        s_vector_substrate = np.moveaxis(s_vector_substrate, -1, 0)
        
    s_vector_substrate = s_vector_substrate[..., np.newaxis]
    s_vector = system_matrix_s @ s_vector_substrate
    Bs, Cs = s_vector[..., 0, 0], s_vector[..., 1, 0]

    # Computes the p-vector
    p_vector_substrate = np.array([np.full_like(eta_substrate_p, 1), eta_substrate_p])
        
    if p_vector_substrate.ndim > 2:
        p_vector_substrate = np.moveaxis(p_vector_substrate, -1, 0)
        
    p_vector_substrate = p_vector_substrate[..., np.newaxis]
    p_vector = system_matrix_p @ p_vector_substrate
    Bp, Cp = p_vector[..., 0, 0], p_vector[..., 1, 0]

    Ys = Cs/Bs
    Yp = Cp/Bp

    rs = (eta_ambient_s - Ys)/(eta_ambient_s + Ys)
    rp = (eta_ambient_p - Yp)/(eta_ambient_p + Yp)

    # phase change on reflection from Chipman, absent from Macleod ch 2
    # Implementing results in spiky phase v.s. wavelength
    # phi_s = -np.arctan(np.imag(eta_substrate_s*(Bs*np.conj(Cs)-np.conj(Bs)*Cs))/(eta_substrate_s**2 * Bs*np.conj(Bs) - Cs * np.conj(Cs)))
    # phi_p = -np.arctan(np.imag(eta_substrate_p*(Bp*np.conj(Cp)-np.conj(Bp)*Cp))/(eta_substrate_p**2 * Bp*np.conj(Bp) - Cp * np.conj(Cp)))

    # rs *= np.exp(-1j*phi_s)
    # rp *= np.exp(-1j*phi_p)

    return rs, rp