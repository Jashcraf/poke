import numpy as np
import poke.poke_core as pol
import matplotlib.pyplot as plt
from astropy.io import fits

pth = 'ELT_Paper_1/ELT/Jones_Pupils/ELT_BareAg_445.fits'
j = fits.open(pth)[0].data

# reclaim the pupil
j00 = j[:,:,0,0,0] + 1j*j[:,:,0,0,1]
j01 = j[:,:,0,1,0] + 1j*j[:,:,0,1,1]
j10 = j[:,:,1,0,0] + 1j*j[:,:,1,0,1]
j11 = j[:,:,1,1,0] + 1j*j[:,:,1,1,1]

diaHV = np.empty([j00.shape[0],j00.shape[1]])
dia45 = np.empty([j00.shape[0],j00.shape[1]])
diaLR = np.empty([j00.shape[0],j00.shape[1]])
retHV = np.empty([j00.shape[0],j00.shape[1]])
ret45 = np.empty([j00.shape[0],j00.shape[1]])
retLR = np.empty([j00.shape[0],j00.shape[1]])

for i in range(j00.shape[0]):
    for j in range(j00.shape[1]):
        J = np.array([[j00[i,j],j01[i,j]],[j10[i,j],j11[i,j]]])
        
        amp,phase,dia,ret = pol.DiattenuationAndRetardancdFromPauli(J)
        
        diaHV[i,j] = dia[0]
        dia45[i,j] = dia[1]
        diaLR[i,j] = dia[2]
        
        retHV[i,j] = ret[0]
        ret45[i,j] = ret[1]
        retLR[i,j] = ret[2]
        

# Time to make a Mueller Pupil
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import numpy as np

vmin = None
vmax = None

titlist = ['DHV','D45','DLR','RHV','R45','RLR']

k = 0

fig = plt.figure(figsize=[12,6])
grid = ImageGrid(fig,111,
                 nrows_ncols=(2,3),
                 axes_pad=0.8,
                cbar_location="right",
                cbar_mode='each',
                direction = 'row',
                cbar_size="5%",
                cbar_pad=0.2)

indx = -1

for ax, im in zip(grid,[diaHV,dia45,diaLR,retHV,ret45,retLR]):
    img = ax.imshow(im,cmap='magma')
    ax.cax.colorbar(img)
    ax.set_title(titlist[k])
    k += 1
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
plt.subplots_adjust()
plt.show()

