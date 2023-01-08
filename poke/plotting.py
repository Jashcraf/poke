import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Goal is to set-up publication-ready plots for PSF's and Jones Pupils

# Default params
params = {
    'image.origin':'lower',
    'image.interpolation':'nearest',
    'image.cmap':'magma',
    'axes.labelsize':20,
    'axes.titlesize':20,
    'font.size':14,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'figure.figsize':[3.39,2.10],
    'font.family':'serif',
}

mpl.rcParams.update(params)

def PlotRayset(rayset_number,xData,yData,lData,mData,surf=-1):

    """Plots ray diagram at a given surface

    Parameters
    ----------
    rayset_number : int

    xData : numpy.ndarray

    yData : numpy.ndarray

    lData : numpy.ndarray

    mData : numpy.ndarray

    surf: int
        Defaults to last surface.
    """

    plt.figure()
    plt.subplot(121)
    plt.title('Position on surface {surface}'.format(surface=surf))
    plt.scatter(xData[rayset_number,surf],yData[rayset_number,surf])
    plt.xlabel('[m]')
    plt.ylabel('[m]')

    plt.subplot(122)
    plt.title('Angle on surface {surface}'.format(surface=surf))
    plt.scatter(lData[rayset_number,surf],mData[rayset_number,surf])
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.show()

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

def PRTPlot(raybundle,surf=0):

    xData = raybundle.xData[surf]
    yData = raybundle.yData[surf]

    if surf == 0:
        Ptot = raybundle.Ptot
    else:
        Ptot = raybundle.P[surf]


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

def PlotJonesPupil(x,y,Jmat,vmin_amp=None,vmax_amp=None,vmin_opd=None,vmax_opd=None):

    fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
    plt.suptitle('|Jones Matrix| for Surface in Hubble')
    for j in range(3):
        for k in range(3):
            ax = axs[j,k]
            ax.set_title('J{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(x,y,c=np.abs(Jmat[:,j,k]),vmin=vmin_amp,vmax=vmax_amp)
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
            sca = ax.scatter(x,y,c=np.angle(Jmat[:,j,k]),vmin=vmin_opd,vmax=vmax_opd)
            fig.colorbar(sca,ax=ax)
    plt.show()
    
def MuellerPupil(M):
    fig,axs = plt.subplots(figsize=[12,12],nrows=4,ncols=4)
    plt.suptitle('Mueller Pupil')
    for i in range(4):
        for j in range(4):
            ax = axs[i,j]
            ax.set_title('J{i}{j}'.format(i=i,j=j))
            sca = ax.imshow(M[i,j,:,:])
            fig.colorbar(sca,ax=ax)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    plt.show()

def PointSpreadMatrix(PSM):
    from matplotlib.colors import LogNorm
    fig,axs = plt.subplots(figsize=[12,12],nrows=4,ncols=4)
    plt.suptitle('Point-spread Matrix')
    for i in range(4):
        for j in range(4):
            ax = axs[i,j]
            ax.set_title('M{i}{j}'.format(i=i,j=j))
            sca = ax.imshow(PSM[...,i,j],cmap='coolwarm')
            fig.colorbar(sca,ax=ax)
            
            if i != 3:
                ax.axes.xaxis.set_visible(False)
            if j != 0:
                ax.axes.yaxis.set_visible(False)
    plt.show()
    
def JonesPupil(raybundle,surf=0):
    x = raybundle.xData[0,0]
    y = raybundle.yData[0,0]
    Jmat = raybundle.JonesPupil[surf]
    
    fig,axs = plt.subplots(figsize=[12,6],nrows=2,ncols=4)
    plt.suptitle('Jones Pupil')
    for j in range(2):
        for k in range(2):
            ax = axs[j,k]
            ax.set_title('|J{j}{k}|'.format(j=j,k=k))
            sca = ax.scatter(x,y,c=np.abs(Jmat[...,j,k]),cmap='inferno')
            fig.colorbar(sca,ax=ax)
            
            # turn off the ticks
            if j != 1:
                ax.xaxis.set_visible(False)
            if k != 0:
                ax.yaxis.set_visible(False)

    for j in range(2):
        for k in range(2):
        
            # Offset the p coefficient
            if j == 1:
                if k == 1:
                    offset = -np.pi
                else:
                    offset = 0
            else:
                offset = 0

            ax = axs[j,k+2]
            ax.set_title(r'$\angle$' + 'J{j}{k}'.format(j=j,k=k))
            sca = ax.scatter(x,y,c=np.angle(Jmat[...,j,k])+offset,cmap='coolwarm')
            fig.colorbar(sca,ax=ax)
            
            # turn off the ticks
            if j != 1:
                ax.xaxis.set_visible(False)
            
            ax.yaxis.set_visible(False)
    plt.show()
    
def AmplitudeResponseMatrix(ARM,lim=None):
    
    from matplotlib.colors import LogNorm
    
    norm = np.max(np.abs(ARM[...,0,0]))
    print('Normalized to Exx intensity of ',norm)
    fig,axs = plt.subplots(figsize=[6,6],nrows=2,ncols=2)
    plt.suptitle('Amplitude Response Matrix')
    for j in range(2):
        for k in range(2):
            ax = axs[j,k]
            ax.set_title('|J{j}{k}|'.format(j=j,k=k))
            sca = ax.imshow(np.abs(ARM[...,j,k])/norm,cmap='inferno',norm=LogNorm(vmax=1,vmin=1e-10),interpolation=None)
            fig.colorbar(sca,ax=ax)
            
            # turn off the ticks
            if j != 1:
                ax.xaxis.set_visible(False)
            if k != 0:
                ax.yaxis.set_visible(False)
                
            # set x,ylim
            if lim != None:
                size = ARM[...,j,k].shape[0]/2
                ax.set_xlim([size-lim,size+lim])
                ax.set_ylim([size-lim,size+lim])
                
    plt.show()

# def JonesPlot(raybundle,surf=-1):

#     x = raybundle.xData[surf]
#     y = raybundle.yData[surf]
#     Jmat = raybundle.J[surf]


#     fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
#     plt.suptitle('|Jones Matrix| for Surface in Hubble')
#     for j in range(3):
#         for k in range(3):
#             ax = axs[j,k]
#             ax.set_title('J{j}{k}'.format(j=j,k=k))
#             sca = ax.scatter(x,y,c=np.abs(Jmat[j,k,:]))
#             fig.colorbar(sca,ax=ax)
#     plt.show()

#     fig,axs = plt.subplots(figsize=[9,9],nrows=3,ncols=3)
#     plt.suptitle('Arg{Jones Matrix} for Surface in Hubble')
#     for j in range(3):
#         for k in range(3):

#             # Offset the p coefficient
#             if j == 1:
#                 if k == 1:
#                     offset = np.pi
#                 else:
#                     offset = 0
#             else:
#                 offset = 0

#             ax = axs[j,k]
#             ax.set_title('J{j}{k}'.format(j=j,k=k))
#             sca = ax.scatter(x,y,c=np.angle(Jmat[j,k,:])+offset)
#             fig.colorbar(sca,ax=ax)
#     plt.show()

# def PlotRays(raybundle):

#     plt.figure(figsize=[12,4])
#     plt.subplot(131)
#     plt.title('Position')
#     plt.scatter(raybundle.xData[0],raybundle.yData[0])

#     plt.subplot(132)
#     plt.title('Direction Cosine')
#     plt.scatter(raybundle.lData[0],raybundle.mData[0])

#     plt.subplot(133)
#     plt.title('Surface Normal Direction Cosine')
#     plt.scatter(raybundle.l2Data[0],raybundle.m2Data[0])
#     plt.show()

# def PlotJonesArray(J11,J12,J21,J22):

#     plt.figure(figsize=[15,7])

#     plt.subplot(241)
#     plt.imshow(np.abs(J11))
#     plt.colorbar()
#     plt.title('J00')

#     plt.subplot(243)
#     plt.imshow(np.angle(J11))
#     plt.colorbar()
#     plt.title('J00')

#     plt.subplot(242)
#     plt.imshow(np.abs(J12))
#     plt.colorbar()
#     plt.title('J01')

#     plt.subplot(244)
#     plt.imshow(np.angle(J12))
#     plt.colorbar()
#     plt.title('J00')



#     plt.subplot(245)
#     plt.imshow(np.abs(J21))
#     plt.colorbar()
#     plt.title('J10')

#     plt.subplot(247)
#     plt.imshow(np.angle(J21))
#     plt.colorbar()
#     plt.title('J10')

#     plt.subplot(246)
#     plt.imshow(np.abs(J22))
#     plt.colorbar()
#     plt.title('J11')

#     plt.subplot(248)
#     plt.imshow(np.angle(J22))
#     plt.colorbar()
#     plt.title('J11')

#     plt.show()



    

