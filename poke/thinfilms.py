import numpy as np

# Inputs are list of index, distance, and the wavelength


def ComputeThinFilmCoeffs(stack,aoi,wavelength):

    # Assemble matrix CLY Equation 13.22
    # CLY relies on determining the AOI in the substrate, which requires a lot of Snells Law calculations, but
    # n*sin(th_in) = n*sin(th_out), so it should all cancel out and just be a function of the aoi. I think the BYU
    # book goes over this
    # What is characteristic admittance?
    nvec = stack[0]
    dvec = stack[1]

    # nvec = [ambient_index,layer1,layer2,...layerN]
    # dvec = [distance1,distance2,...distanceN]

    k = 2*np.pi/wavelength # Does this need to include the index? I

    # snell's law to the final media angle of incidence, because we multiply backward
    aon = -np.arcsin(nvec[0]*np.sin(aoi)/(nvec[-1]))

    # final matrices
    Mp_final = np.array([[np.cos(aon),0],[nvec[-1],0]])
    Ms_final = np.array([[1,0],[nvec[-1]*np.cos(aon),0]])

    # initial matrices
    Mp = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0],np.cos(aoi)],[nvec[0],-np.cos(aoi)]])
    Ms = 1/(2*nvec[0]*np.cos(aoi)) * np.array([[nvec[0]*np.cos(aoi),1],[nvec[0]*np.cos(aoi),-1]])

    # now shed the prepended ambient index
    # nvec = nvec[1:]

    ## Create Characteristic Matrix

    # Loop over numcoatings
    for q in range(len(nvec)-1):

        if q == 0:
            aor = aoi
        
        # Snells Law to next surface (the new aoi)
        arg = np.real((nvec[q])*np.sin(aor)/(nvec[q+1]))
        # B.append()
        aor = np.arcsin(arg)
        
        # update B calculation
        B = k*nvec[q+1]*dvec[q]*np.cos(aor)
        # print(B)

        # print(nvec[q])
        # print(nvec[q+1])

        # Need to multiply through the stack
        Mp = Mp @ np.array([[np.cos(B),-1j*np.sin(B)*np.cos(aor)/nvec[q+1]],
                            [-1j*nvec[q+1]*np.sin(B)/np.cos(aor),np.cos(B)]])

        Ms = Ms @ np.array([[np.cos(B),-1j*np.sin(B)/(np.cos(aor)*nvec[q+1])],
                            [-1j*nvec[q+1]*np.cos(aor)*np.sin(B),np.cos(B)]])

    # Now multiply by the scalar from the first layer
    # B.append(1)
    Ap =  Mp @ Mp_final
    As =  Ms @ Ms_final

    # print(Ap)
    # print(As)

    tp = 1/Ap[0,0]
    rp = Ap[1,0]/Ap[0,0]
    ts = 1/As[0,0]
    rs = As[1,0]/As[0,0]

    return tp,rp,ts,rs

    
def TwoLayerThinFilms(nvec,d,aoi,wavelength):

    # nvec is a 2-vector of refractive indices

    # d is a film thickness

    # aoi is in radians

    return