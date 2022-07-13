import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

params = {
    'image.origin':'lower',
    'image.interpolation':'nearest',
    'image.cmap':'magma',
    'axes.labelsize':12,
    'axes.titlesize':12,
    'font.size':8,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.figsize':[3.39,2.10],
    'font.family':'serif',
}

mpl.rcParams.update(params)

def AOIPlot(raybundle,surf=-1,units='degrees'):

    xData = raybundle.xData[surf]
    yData = raybundle.yData[surf]
    aoi = raybundle.aoi[surf]

    if units == 'degrees':
        aoi *= 180/np.pi

    plt.figure()
    plt.title('AOI [{uni}] on surface {surface}'.format(uni=units,surface=surf))
    plt.scatter(xData,yData,c=aoi)
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.colorbar()
    plt.show()
    