import numpy as np

# Inputs are list of index, distance, and the wavelength

def ComputeThinFilmCoeffs(nvec,darray,aoi,wavelength):

    # Assemble matrix CLY Equation 13.22
    # CLY relies on determining the AOI in the substrate, which requires a lot of Snells Law calculations, but
    # n*sin(th_in) = n*sin(th_out), so it should all cancel out and just be a function of the aoi. I think the BYU
    # book goes over this
    # What is characteristic admittance?

    k = 2*np.pi/wavelength # Does this need to include the index?

    aon = np.arcsin(nvec[0]*np.sin(aoi[0])/nvec[-1])

    Mp = np.array([[np.cos(aon),0],[nvec[-1],0]])
    Ms = np.array([[1,0],[nvec[-1]*np.cos(aon),0]])

    ## Create Characteristic Matrix

    # Loop over numcoatings
    for q in range(len(nvec)):

        # Loop over distances
        for p in range(len(darray)):

            B = k*darray[p]*np.cos(aoi[p])

            # Snells Law to next surface
            aor = np.arcsin(nvec[q]*np.sin(aoi[p])/nvec[q+1])

            # Need to reverse-multiply through the stack
            Mp = np.array([[np.cos(B),-1j*np.sin(B)*np.cos(aoi[p])/nvec[q]],
                           [-1j*nvec[q]*np.sin(B)/np.cos(aoi[p]),np.cos(B)]])@ Mp

            Ms = np.array([[np.cos(B),-1j*np.sin(B)/(np.cos(aoi[p])*nvec[q])],
                          [-1j*nvec[q]*np.cos(aoi[p])*np.sin(B),np.cos(B)]])@ Ms

    # Now compute the influence from the first layer, which I think is the air layer?
    Ap = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0],np.cos(aoi)],[nvec[0],-np.cos(aoi)]]) @ Mp
    As = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0]*np.cos(aoi),1],[nvec[0]*np.cos(aoi),-1]]) @ Ms

    tp = 1/Ap[0,0]
    rp = Ap[1,0]/Ap[0,0]
    ts = 1/As[0,0]
    rs = As[1,0]/As[0,0]

    return tp,rp,ts,rs

    
def TwoLayerThinFilms(nvec,dvec,aoi,wavelength):

    return