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

def PRTPlot(raybundle,surf=-1):

    xData = raybundle.xData[surf]
    yData = raybundle.yData[surf]
    Ptot = raybundle.Pmat

    fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
    plt.suptitle('|PRT Matrix| for System')
    for j in range(3):
        for k in range(3):
            ax = axs[j,k]
            ax.set_title('P{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(xData,yData,c=np.abs(Ptot[j,k,:]))
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.colorbar(sca,ax=ax)
    plt.show()

    fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
    plt.suptitle('Arg(PRT Matrix) for System')
    for j in range(3):
        for k in range(3):
            ax = axs[j,k]
            ax.set_title('P{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(xData,yData,c=np.angle(Ptot[j,k,:]))
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.colorbar(sca,ax=ax)
    plt.show()
    
def JonesPlot(raybundle,surf=-1):

    x = raybundle.xData[surf]
    y = raybundle.yData[surf]
    Jmat = raybundle.Jmat


    fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
    plt.suptitle('|Jones Matrix| for Surface in Hubble')
    for j in range(3):
        for k in range(3):
            ax = axs[j,k]
            ax.set_title('J{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(x,y,c=np.abs(Jmat[j,k,:]))
            fig.colorbar(sca,ax=ax)
    plt.show()

    fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
    plt.suptitle('Arg{Jones Matrix} for Surface in Hubble')
    for j in range(3):
        for k in range(3):

            # Offset the p coefficient
            if j == 1:
                if k == 1:
                    offset = np.pi
                else:
                    offset = 0
            else:
                offset = 0

            ax = axs[j,k]
            ax.set_title('J{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(x,y,c=np.angle(Jmat[j,k,:])+offset)
            fig.colorbar(sca,ax=ax)
    plt.show()

def PlotRays(raybundle):

    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.title('Position')
    plt.scatter(raybundle.xData[0],raybundle.yData[0])

    plt.subplot(132)
    plt.title('Direction Cosine')
    plt.scatter(raybundle.lData[0],raybundle.mData[0])

    plt.subplot(133)
    plt.title('Surface Normal Direction Cosine')
    plt.scatter(raybundle.l2Data[0],raybundle.m2Data[0])
    plt.show()

    
