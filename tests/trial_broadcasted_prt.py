import numpy as np
from poke.poke_core import Rayfront
import poke.raytrace as ray
import time
import matplotlib.pyplot as plt
import poke.plotting as plot
import pickle
from astropy.io import fits

pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/tests/Hubble_Test.zmx'
nrays = 24
wave = 1
global_coords = True

n_Al = 1.2 - 1j*7.115 # 600nm from CV Al coating MUL
wvl = 600e-9

s1 = {
    'surf':2,
    'coating':n_Al,
    'mode':'reflect'
}

s2 = {
    'surf':4,
    'coating':n_Al,
    'mode':'reflect'
}

s3 = {
    'surf':8,
    'coating':n_Al,
    'mode':'reflect'
}

surflist = [s1,s2,s3]

def plot3x3(raybundle,op=np.abs):
    """plots a 3x3 matrix"""

    x = raybundle.xData[0,0]
    y = raybundle.yData[0,0]

    fig,ax = plt.subplots(nrows=3,ncols=3)
    for row in range(3):
        for column in range(3):

            ax[row,column].scatter(x,y,c=op(raybundle.P_total[0][...,row,column]))
    plt.show()

if __name__ == '__main__':

    tlist_old = []
    tlist_new = []
    a = np.array([0,1.,0.])
    x = np.array([1.,0.,0.])

    rays = 32

    rf = Rayfront(nrays,wvl,1.2,.08)
    rf.as_polarized(surflist)
    rf.trace_rayset(pth)

    # rf_old = Rayfront(nrays,wvl,1.2,.08)
    # rf_old.as_polarized(surflist)
    # rf_old.trace_rayset(pth)

    rf.compute_jones_pupil(aloc=a,exit_x=x)
    jones = rf.jones_pupil

    op = np.abs
    cmap = 'magma'
    fig,ax = plt.subplots(ncols=3,nrows=3)
    for j in range(3):
        for i in range(3):
            sca = ax[i,j].scatter(rf.xData[0][0],rf.yData[0][0],c=op(jones[0][...,i,j]),cmap=cmap)
            fig.colorbar(sca,ax=ax[i,j])
    plt.show()

    op = np.angle
    cmap = 'RdBu'
    fig,ax = plt.subplots(ncols=3,nrows=3)
    for j in range(3):
        for i in range(3):
            sca = ax[i,j].scatter(rf.xData[0][0],rf.yData[0][0],c=op(jones[0][...,i,j]),cmap=cmap)
            fig.colorbar(sca,ax=ax[i,j])
    plt.show()

    # t1_old = time.perf_counter()
    # rf_old.ComputeJonesPupil(aloc=a,exit_x=x)
    # tlist_old.append(time.perf_counter()-t1_old)

    # plt.style.use('fivethirtyeight')
    # plt.figure()
    # plt.plot(rays,tlist_new,label='vectorized')
    # plt.plot(rays,tlist_old,label='looped')
    # plt.legend()
    # plt.xlabel('Number of rays along pupil edge')
    # plt.ylabel('Time to compute Jones pupil [s]')
    # plt.title('Runtime comparison of algorithms')
    # plt.show()

    # plt.style.use('fivethirtyeight')
    # plt.figure()
    # plt.plot(rays,tlist_new,label='vectorized')
    # plt.plot(rays,tlist_old,label='looped')
    # plt.legend()
    # plt.xlabel('Number of rays along pupil edge')
    # plt.ylabel('Time to compute Jones pupil [s]')
    # plt.title('Runtime comparison of algorithms')
    # plt.yscale('log')
    # plt.show()

    