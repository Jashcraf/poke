import numpy as np

# Inputs are list of index, distance, and the wavelength

def ComputeThinFilmCoeffs(nvec,dvec,aoi,wavelength):

    # Assemble matrix CLY Equation 13.22
    # CLY relies on determining the AOI in the substrate, which requires a lot of Snells Law calculations, but
    # n*sin(th_in) = n*sin(th_out), so it should all cancel out and just be a function of the aoi. I think the BYU
    # book goes over this
    # What is characteristic admittance?

    for q in range(len(nvec)):

        # Compute beta
        Bq = 2*np.pi*nvec[q]*dvec[q]*np.cos(aoi)

    
    