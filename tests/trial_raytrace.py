import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
from astropy.io import fits

pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/tests/Hubble_Test.zmx'
nrays = 6
wave = 1
global_coords = True

n_Al = 1.2 + 1j*7.115 # 600nm from CV Al coating MUL
n_Al = n_Al #np.complex64(n_Al)
wvl = 600e-9

s1 = {
    'surf':2,
    'coating':n_Al,
    'mode':'reflect'
}

s2 = {
    'surf':3,
    'coating':n_Al,
    'mode':'reflect'
}

s3 = {
    'surf':4,
    'coating':n_Al,
    'mode':'reflect'
}

surflist = [s1,s2,s3]

# def test_TraceThroughCV(nrays):

#     rayset = np.array([np.random.rand(nrays),
#                        np.random.rand(nrays),
#                        np.random.rand(nrays),
#                        np.random.rand(nrays)])   # random ray in the first pupil and field quadrant
#     raysets = [rayset]

#     raydata = ray.TraceThroughCV(raysets,pth,surflist,nrays,wave,global_coords)

#     return raydata

def test_TraceRayfrontThroughCV(nrays):

    raybundle = Rayfront(nrays,wvl,1.2,0.08,normalized_pupil_radius=1,fov=[0.,0.],circle=True)
    raybundle.as_polarized(surflist)
    print(raybundle.global_coords)
    raybundle.trace_rayset(pth)

    return raybundle

if __name__ == '__main__':

    raybundle_cv = test_TraceRayfrontThroughCV(nrays)
        
    # with open('Hubble_Test_RayfrontCV.pickle','rb') as f:
    #     raybundle_cv = pickle.load(f)

    # with open('Hubble_Test_RayfrontZMX.pickle','rb') as n:
    #     raybundle_zmx = pickle.load(n)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # for i in range(3):
    # i = 1
    # ax.quiver(raybundle_cv.xData[0][i],raybundle_cv.yData[0][i],raybundle_cv.zData[0][i],
    #             raybundle_cv.lData[0][i],raybundle_cv.mData[0][i],raybundle_cv.nData[0][i],
    #             length=5e-3,normalize=True,label=f'CV surf {i}',color='black',alpha=0.5)
    # ax.quiver(raybundle_zmx.xData[0][i],raybundle_zmx.yData[0][i],raybundle_zmx.zData[0][i],
    #             raybundle_zmx.lData[0][i],raybundle_zmx.mData[0][i],raybundle_zmx.nData[0][i],
    #             length=5e-3,normalize=True,label=f'ZMX surf {i}',color='red',alpha=0.5)
    #     # ax.quiver(raybundle_zmx.xData[0][i],raybundle_zmx.yData[0][i],raybundle_zmx.zData[0][i],label=f'ZMX surf {i}')
    # plt.legend()
    # plt.show()

    raybundle_cv.ComputeJonesPupil(aloc=np.array([0.,1.,0.]))
    jones = raybundle_cv.JonesPupil
    jones_real = np.real(jones)
    jones_imag = np.imag(jones)
    
    def write_to_fits(array,fn):
        hdu = fits.PrimaryHDU(array)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fn,overwrite=True)

    write_to_fits(jones_real,'hst_fold_real_wl600.fits')
    write_to_fits(jones_imag,'hst_fold_imag_wl600.fits')

    # raybundle_zmx.ComputeJonesPupil(aloc=np.array([0.,1.,0.]))
    plot.JonesPupil(raybundle_cv)
    # plot.JonesPupil(raybundle_zmx)
    
    # sind = 1
    # xdiff = raybundle_cv.xData[0][sind] - raybundle_zmx.xData[0][sind]
    # ydiff = raybundle_cv.yData[0][sind] - raybundle_zmx.yData[0][sind]
    # zdiff = raybundle_cv.zData[0][sind] - raybundle_zmx.zData[0][sind]
    
    # ldiff = raybundle_cv.lData[0][sind] - raybundle_zmx.lData[0][sind]
    # mdiff = raybundle_cv.mData[0][sind] - raybundle_zmx.mData[0][sind]
    # ndiff = raybundle_cv.nData[0][sind] - raybundle_zmx.nData[0][sind]

    # plt.figure()
    # plt.subplot(161)
    # plt.title('X')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=xdiff)
    # plt.colorbar()
    # plt.subplot(162)
    # plt.title('Y')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=ydiff)
    # plt.colorbar()
    # plt.subplot(163)
    # plt.title('Z')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=zdiff)
    # plt.colorbar()
    # plt.subplot(164)
    # plt.title('L')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=ldiff)
    # plt.colorbar()
    # plt.subplot(165)
    # plt.title('M')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=mdiff)
    # plt.colorbar()
    # plt.subplot(166)
    # plt.title('N')
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=ndiff)
    # plt.colorbar()
    # plt.show()

    # Compute Jones Pupils and compare
    # raybundle_zmx.ComputeJonesPupil(aloc=np.array([0.,1.,0.]))
    # raybundle_cv.ComputeJonesPupil(aloc=np.array([0.,1.,0.]))
    # plot.JonesPupil(raybundle_zmx)
    # plot.JonesPupil(raybundle_cv)


    # timelist = []

    # for i in range(5):
    #     nrays = int(10*(i+1))**2
    #     t1 = time.perf_counter()
    #     data = test_TraceThroughCV(nrays)
    #     ttot = time.perf_counter()-t1
    #     print(f'time for {nrays} rays = {ttot}s')
    #     timelist.append(ttot)

    # plt.figure()
    # plt.plot(np.arange(10,50,10),timelist)
    # # plt.errorbar(np.arange(10,50,10),timelist,yerr=np.std(timelist))
    # plt.xlabel('nrays across pupil')
    # plt.ylabel('runtime [s]')
    # plt.title('Timing poke.raytrace.TraceThroughCV')
    # plt.show()
