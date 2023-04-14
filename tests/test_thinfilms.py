# Tests the polarization functions
import poke.thinfilms as tf
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Refractive index v.s. wavelength data from refractiveindex.info
wlens = [600e-9,650e-9,700e-9,750e-9,800e-9]
n_BK7 = [1.5163,1.5145,1.5131,1.5118,1.5108] 
n_SiO2 = [1.4580,1.4565,1.4553,1.4542,1.4533]
n_Al2O3 = [1.7675,1.7651,1.7632,1.7616,1.7601 + 0*1j]
d_film = 25e-9
npts_to_test = 10
aoi_deg = 45
aoi = np.full(npts_to_test,aoi_deg * np.pi/180)

def test_compute_thin_films_broadcasted(plotfilms=False):
    """tests the thin film algorithm, looped & vectorized, against data from filmetrics
    https://www.filmetrics.com/reflectance-calculator

    This is currently limited by our ability to get a thin film model and re-create it's refractive index data exactly,
    But this test matches to 1e-3 of the fresnel coefficient
    """

    # Pre-allocate result arrays
    Rs_looped = np.empty([npts_to_test,len(wlens)])
    Rp_looped = np.copy(Rs_looped)
    Rs_broad= np.copy(Rs_looped)
    Rp_broad= np.copy(Rs_looped)


    # from filmetrics
    # Rs_45_pth = 'poke/tests/Reflectance-Calcs-Rs.txt'
    # Rp_45_pth = 'poke/tests/Reflectance-Calcs-Rp.txt'
    # Rs_45_ref = np.genfromtxt(Rs_45_pth,skip_header=1)[0:5,1]
    # Rp_45_ref = np.genfromtxt(Rp_45_pth,skip_header=1)[0:5,1]
    Rs_45_ref = [0.133723,0.1318435,0.1277324,0.1253954,0.1230782]
    Rp_45_ref = [0.0185438,0.0179579,0.0168185,0.0162043,0.0156147]
    # Generate the looped results
    for i,wlen in enumerate(wlens):
        # define a static film
        stack = [
            (n_BK7[i],d_film),
            (n_SiO2[i],d_film),
            (n_Al2O3[i],d_film),
            (n_BK7[i]) # gets ignored here, but not in PRT
        ]
        for j in range(npts_to_test):

            rs,_ = tf.compute_thin_films_broadcasted(stack,np.radians(aoi_deg),wlen,substrate_index=n_BK7[i],polarization='s')
            rp,_ = tf.compute_thin_films_broadcasted(stack,np.radians(aoi_deg),wlen,substrate_index=n_BK7[i],polarization='p')
            Rs_looped[j,i] = np.abs(rs)**2
            Rp_looped[j,i] = np.abs(rp)**2

    # Generate the broadcasted results, still looping over wavelength
    for i,wlen in enumerate(wlens):
        # And define a stack with spatial variance
        stack_broad = [
            (n_BK7[i]*np.ones_like(aoi),d_film*np.ones_like(aoi)),
            (n_SiO2[i]*np.ones_like(aoi),d_film*np.ones_like(aoi)),
            (n_Al2O3[i]*np.ones_like(aoi),d_film*np.ones_like(aoi)),
            (n_BK7[i]) # gets ignored here, but not in PRT
        ]
        rs_broad,_ = tf.compute_thin_films_broadcasted(stack_broad,aoi,wlen,substrate_index=n_BK7[i],polarization='s')
        rp_broad,_ = tf.compute_thin_films_broadcasted(stack_broad,aoi,wlen,substrate_index=n_BK7[i],polarization='p')
        Rs_broad[:,i] = np.abs(rs_broad)**2
        Rp_broad[:,i] = np.abs(rp_broad)**2

    # select the first index from the bunch
    index = 5
    Rs_looped = Rs_looped[index]
    Rp_looped = Rp_looped[index]
    Rs_broad = Rs_broad[index]
    Rp_broad = Rp_broad[index]

    # if plotfilms:
    #     plt.style.use('fivethirtyeight')
    #     plt.figure()
    #     plt.suptitle('Fresnel Reflectance at 45deg')
    #     plt.subplot(121)
    #     plt.plot(wlens,Rs_45_ref,label='Filmetrics Rs')
    #     plt.plot(wlens,Rs_looped,label='Looped Rs')
    #     plt.plot(wlens,Rs_broad,label='Broadcast Rs',linestyle='dashed')
    #     plt.xlabel('wavelengths')
    #     plt.ylabel('Reflectance')
    #     plt.legend()
    #     plt.subplot(122)
    #     plt.plot(wlens,Rp_45_ref,label='Filmetrics Rp')
    #     plt.plot(wlens,Rp_looped,label='Looped Rp')
    #     plt.plot(wlens,Rp_broad,label='Broadcast Rp',linestyle='dashed')
    #     plt.xlabel('wavelengths')
    #     plt.legend()
    #     plt.show()
        

    np.testing.assert_allclose((Rs_looped,Rp_looped,Rs_broad,Rp_broad),(Rs_45_ref,Rp_45_ref,Rs_45_ref,Rp_45_ref),atol=1e-3)



if __name__ == '__main__':

    test_compute_thin_films_broadcasted(plotfilms=True)
    


