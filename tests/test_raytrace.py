import numpy as np
import poke
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt

pth = 'C:/Users/ashcraft/Desktop/poke/test_files/hubble_test.len'
nrays = 11
wave = 1
global_coords = True

s1 = {
    'surf':1,
    'coating':1 + 1j*10,
    'mode':'reflect'
}

s2 = {
    'surf':2,
    'coating':1 + 1j*10,
    'mode':'reflect'
}

surflist = [s1,s2]

def test_TraceThroughCV(nrays):

    rayset = np.array([np.random.rand(nrays),
                       np.random.rand(nrays),
                       np.random.rand(nrays),
                       np.random.rand(nrays)])   # random ray in the first pupil and field quadrant
    raysets = [rayset]

    raydata = ray.TraceThroughCV(raysets,pth,surflist,nrays,wave,global_coords)

    return raydata

if __name__ == '__main__':

    timelist = []

    for i in range(5):
        nrays = int(10*(i+1))**2
        t1 = time.perf_counter()
        data = test_TraceThroughCV(nrays)
        ttot = time.perf_counter()-t1
        print(f'time for {nrays} rays = {ttot}s')
        timelist.append(ttot)

    plt.figure()
    plt.plot(np.arange(10,50,10),timelist)
    # plt.errorbar(np.arange(10,50,10),timelist,yerr=np.std(timelist))
    plt.xlabel('nrays across pupil')
    plt.ylabel('runtime [s]')
    plt.title('Timing poke.raytrace.TraceThroughCV')
    plt.show()