import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
import pickle

pth = 'C:/Users/ashcraft/Desktop/poke/tests/hubble_test.len'
nrays = 30
wave = 1
global_coords = True

s1 = {
    'surf':2,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

s2 = {
    'surf':3,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

s3 = {
    'surf':4,
    'coating':0.04 + 1j*7.1155,
    'mode':'reflect'
}

surflist = [s1,s2,s3]

def test_TraceThroughCV(nrays):

    rayset = np.array([np.random.rand(nrays),
                       np.random.rand(nrays),
                       np.random.rand(nrays),
                       np.random.rand(nrays)])   # random ray in the first pupil and field quadrant
    raysets = [rayset]

    raydata = ray.TraceThroughCV(raysets,pth,surflist,nrays,wave,global_coords)

    return raydata

def test_TraceRayfrontThroughCV(nrays):

    raybundle = Rayfront(nrays,1e-6,1.2,0.08,normalized_pupil_radius=0.9,fov=[0.,0.],circle=True)
    raybundle.as_polarized(surflist)
    print(raybundle.global_coords)
    raybundle.TraceRaysetCV(pth)

    return raybundle

if __name__ == '__main__':

    raybundle_cv = test_TraceRayfrontThroughCV(nrays)

    with open('Hubble_Test_RayfrontCV.pickle','wb') as f:
        pickle.dump(raybundle_cv,f)

    with open('Hubble_Test_Rayfront.pickle','rb') as n:
        raybundle_zmx = pickle.load(n)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(3):
        ax.scatter(raybundle_cv.xData[0][i],raybundle_cv.yData[0][i],raybundle_cv.zData[0][i],label=f'CV surf {i}')
        ax.scatter(raybundle_zmx.xData[0][i],raybundle_zmx.yData[0][i],raybundle_zmx.zData[0][i],label=f'ZMX surf {i}')
    plt.legend()
    plt.show()

    raybundle_cv.ComputeJonesPupil(aloc=np.array([0.,-1.,0.]))
    plot.JonesPupil(raybundle_cv)
    
    # sind = 2

    # xdiff = raybundle_cv.xData[0][sind] - raybundle_zmx.xData[0][sind]
    # ydiff = raybundle_cv.yData[0][sind] - raybundle_zmx.yData[0][sind]
    # zdiff = raybundle_cv.zData[0][sind] - raybundle_zmx.zData[0][sind]
    
    # ldiff = raybundle_cv.lData[0][sind] - raybundle_zmx.lData[0][sind]
    # mdiff = raybundle_cv.mData[0][sind] - raybundle_zmx.mData[0][sind]
    # ndiff = raybundle_cv.nData[0][sind] - raybundle_zmx.nData[0][sind]

    # plt.figure()
    # plt.subplot(161)
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=xdiff)
    # plt.colorbar()
    # plt.subplot(162)
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=ydiff)
    # plt.colorbar()
    # plt.subplot(163)
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=zdiff)
    # plt.colorbar()
    # plt.subplot(164)
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=ldiff)
    # plt.colorbar()
    # plt.subplot(165)
    # plt.scatter(raybundle_zmx.xData[0][0],raybundle_zmx.yData[0][0],c=mdiff)
    # plt.colorbar()
    # plt.subplot(166)
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