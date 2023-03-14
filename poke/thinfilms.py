import numpy as np

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

def ComputeThinFilmCoeffsCLY(stack,aoi,wavelength,vacuum_index=1,substrate_index=1.5):

    """CLY S13.3.1 Algorithms, Macleod 1969 The reflectance of a simple boundary

    TODO - set up to broadcast the matrix multiplication across matmul?
    TODO - check order of system matrix calc

    Parameters
    ----------

    stack : list of tuples
        list composed of elements containing the index (n) and thickness (t) in meters, ordered like
        stack = [(n0,t0),(n1,t1)...,(nN,tN)]
        Where the ambient index is assumed to be unity
    
    aoi : float
        angle of incidence in radians on the thin film stack

    wavelength: float
        wavelength to comput the reflection coefficients for in meters
    
    """

    # Init boundary, this changes on loop iteration
    n1 = vacuum_index

    # Pre-allocate the system matrix
    system_matrix_s = np.array([[ONE_COMPLEX,ZERO_COMPLEX],[ZERO_COMPLEX,ONE_COMPLEX]])
    system_matrix_p = np.array([[ONE_COMPLEX,ZERO_COMPLEX],[ZERO_COMPLEX,ONE_COMPLEX]])

    # Transform to substrate aor in the substrate
    aor = np.arcsin(n1*np.sin(aoi)/substrate_index)

    # Characteristic admittance of the substrate
    eta_medium_s =  FREESPACE_IMPEDANCE_INV * substrate_index * np.cos(aor)
    eta_medium_p =  FREESPACE_IMPEDANCE_INV * substrate_index / np.cos(aor)

    # Characteristic admittancd of free space
    eta0_s = FREESPACE_IMPEDANCE_INV * vacuum_index * np.cos(aoi)
    eta0_p = FREESPACE_IMPEDANCE_INV * vacuum_index / np.cos(aoi)

    eta_vec_s = np.array([1,eta_medium_s])
    eta_vec_p = np.array([1,eta_medium_p])

    for i,film in enumerate(stack):

        # Snells law to angle of wave vector in the current film medium
        aoi = np.arcsin(n1*np.sin(aoi)/film[0])
        # print('aoi = ',aoi)
        # print('n1 = ',n1)

        # Phase thickness of film
        B = 2*np.pi * film[0] * film[1] * np.cos(aoi) / wavelength

        # Characteristic Admittance, s
        eta_s = FREESPACE_IMPEDANCE_INV * film[0] * np.cos(aoi)

        # Characteristic Matrix, s
        characteristic_matrix_s = np.array([[np.cos(B),1j*np.sin(B)/eta_s],
                                            [1j*eta_s*np.sin(B),np.cos(B)]])

        # Characteristic Admittance, p
        eta_p = FREESPACE_IMPEDANCE_INV * film[0] / np.cos(aoi)

        # Characteristic Matrix, p
        characteristic_matrix_p = np.array([[np.cos(B),1j*np.sin(B)/eta_p],
                                            [1j*eta_p*np.sin(B),np.cos(B)]])

        # Add to system matrix, order matters here! TODO - check order of system matrix calc
        # system_matrix_s = characteristic_matrix_s @ system_matrix_s
        # system_matrix_p = characteristic_matrix_p @ system_matrix_p
        system_matrix_s = system_matrix_s @ characteristic_matrix_s 
        system_matrix_p = system_matrix_p @ characteristic_matrix_p

        # Update prior film index to the current, then continue the loop
        n1 = film[0]

    characteristic_vector_s = system_matrix_s @ eta_vec_s
    characteristic_vector_p = system_matrix_p @ eta_vec_p

    # Translate to the Text
    Bs,Cs = characteristic_vector_s[0],characteristic_vector_s[1]
    Bp,Cp = characteristic_vector_p[0],characteristic_vector_p[1]
    # Bs,Cs = eta_vec_s[0],eta_vec_s[1]
    # Bp,Cp = eta_vec_p[0],eta_vec_p[1]

    # s coefficients
    rs = (eta0_s*Bs - Cs)/(eta0_s*Bs + Cs)
    Ps = np.imag(eta_medium_s * (Bs * np.conj(Cs) - Cs * np.conj(Bs))) # add phase change
    Ps /= eta_medium_s**2 * Bs * np.conj(Bs) - Cs * np.conj(Cs)
    rs *= np.exp(-1j*np.arctan(Ps))

    ts = np.conj(2*eta0_s/(eta0_s*Bs + Cs))

    # p coefficients
    rp = (eta0_p*Bp - Cp)/(eta0_p*Bp + Cp)
    Pp = np.imag(eta_medium_p * (Bp * np.conj(Cp) - Cp * np.conj(Bp))) # add phase change
    Pp /= eta_medium_p**2 * Bp * np.conj(Bp) - Cp * np.conj(Cp)
    rp *= np.exp(-1j*np.arctan(Pp))
    tp = np.conj(2*eta0_p/(eta0_p*Bp + Cp))
    

    return rs,ts,rp,tp




# def ComputeThinFilmCoeffs(stack,aoi,wavelength):

#     # Assemble matrix CLY Equation 13.22
#     # CLY relies on determining the AOI in the substrate, which requires a lot of Snells Law calculations, but
#     # n*sin(th_in) = n*sin(th_out), so it should all cancel out and just be a function of the aoi. I think the BYU
#     # book goes over this
#     # What is characteristic admittance?
#     nvec = stack[0]
#     dvec = stack[1]

#     # nvec = [ambient_index,layer1,layer2,...layerN]
#     # dvec = [distance1,distance2,...distanceN]

#     k = 2*np.pi/wavelength # Does this need to include the index? I

#     # snell's law to the final media angle of incidence, because we multiply backward
#     aon = -np.arcsin(nvec[0]*np.sin(aoi)/(nvec[-1]))

#     # final matrices
#     Mp_final = np.array([[np.cos(aon),0],[nvec[-1],0]])
#     Ms_final = np.array([[1,0],[nvec[-1]*np.cos(aon),0]])

#     # initial matrices
#     Mp = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0],np.cos(aoi)],[nvec[0],-np.cos(aoi)]])
#     Ms = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0]*np.cos(aoi),1],[nvec[0]*np.cos(aoi),-1]])

#     # now shed the prepended ambient index
#     # nvec = nvec[1:]

#     ## Create Characteristic Matrix

#     # Loop over numcoatings
#     for q in range(len(nvec)-1):

#         if q == 0:
#             aor = aoi
        
#         # Snells Law to next surface (the new aoi)
#         arg = np.real((nvec[q])*np.sin(aor)/(nvec[q+1]))
#         # B.append()
#         aor = np.arcsin(arg)
        
#         # update B calculation
#         B = k*nvec[q+1]*dvec[q]*np.cos(aor)
#         # print(B)

#         # print(nvec[q])
#         # print(nvec[q+1])

#         # Need to multiply through the stack
#         Mp = Mp @ np.array([[np.cos(B),-1j*np.sin(B)*np.cos(aor)/nvec[q+1]],
#                             [-1j*nvec[q+1]*np.sin(B)/np.cos(aor),np.cos(B)]])

#         Ms = Ms @ np.array([[np.cos(B),-1j*np.sin(B)/(np.cos(aor)*nvec[q+1])],
#                             [-1j*nvec[q+1]*np.cos(aor)*np.sin(B),np.cos(B)]])

#     # Now multiply by the scalar from the first layer
#     # B.append(1)
#     Ap =  Mp @ Mp_final
#     As =  Ms @ Ms_final

#     # print(Ap)
#     # print(As)

#     tp = 1/Ap[0,0]
#     rp = Ap[1,0]/Ap[0,0]
#     ts = 1/As[0,0]
#     rs = As[1,0]/As[0,0]

#     return tp,rp,ts,rs

    
# def TwoLayerThinFilms(nvec,d,aoi,wavelength):

#     # nvec is a 2-vector of refractive indices

#     # d is a film thickness

#     # aoi is in radians

#     return