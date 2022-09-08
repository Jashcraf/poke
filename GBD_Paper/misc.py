# Miscellaneous functions for creating plots and saving python objects with pickle

import numpy as np
try:
    import cupy as cp
    cupy_available = True
except:
    cupy_available = False
    import numpy as cp
#import proper 
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy
import astropy.io.fits as fits
import astropy.units as u
import pickle
import matplotlib
import copy

def myimshow(arr, title=None, 
             npix=None,
             lognorm=False, vmin=None, vmax=None,
             cmap='magma',
             pxscl=None,
             patches=None,
             figsize=(4,4), dpi=125, display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if cupy_available and isinstance(arr, cp.ndarray):
        arr = arr.get()
    
    if npix is not None:
        arr = pad_or_crop(arr, npix)
    
    if pxscl is not None:
        if isinstance(pxscl, u.Quantity):
            if pxscl.unit==(u.meter/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('meters')
            elif pxscl.unit==(u.mm/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('millimeters')
            elif pxscl.unit==(u.arcsec/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('arcsec')
            elif pxscl.unit==(u.mas/u.pix):
                vext = pxscl.value * arr.shape[0]/2
                hext = pxscl.value * arr.shape[1]/2
                extent = [-vext,vext,-hext,hext]
                ax.set_xlabel('mas')
        else:
            vext = pxscl * arr.shape[0]/2
            hext = pxscl * arr.shape[1]/2
            extent = [-vext,vext,-hext,hext]
            ax.set_xlabel('lambda/D')
    else:
        extent=None
    
    if lognorm:
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = Normalize(vmin=vmin,vmax=vmax)
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
            
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    
def myimshow2(arr1, arr2, 
              title1=None, title2=None,
              npix=None, npix1=None, npix2=None,
              pxscl=None, pxscl1=None, pxscl2=None,
              cmap1='magma', cmap2='magma',
              lognorm1=False, lognorm2=False,
              vmin1=None, vmax1=None, vmin2=None, vmax2=None, 
              patches1=None, patches2=None,
              display_fig=True, 
              return_fig=False, 
              figsize=(10,4), dpi=125, wspace=0.2):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    if cp and isinstance(arr1, cp.ndarray):
        arr1 = arr1.get()
    if cp and isinstance(arr2, cp.ndarray):
        arr2 = arr2.get()
    
    if npix is not None:
        arr1 = pad_or_crop(arr1, npix)
        arr2 = pad_or_crop(arr2, npix)
    if npix1 is not None:
        arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None:
        arr2 = pad_or_crop(arr2, npix2)
    
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            if pxscl1.unit==(u.meter/u.pix): ax[0].set_xlabel('meters')
            elif pxscl1.unit==(u.millimeter/u.pix): ax[0].set_xlabel('millimeters')
            elif pxscl1.unit==(u.arcsec/u.pix): ax[0].set_xlabel('arcsec')
            elif pxscl1.unit==(u.mas/u.pix): ax[0].set_xlabel('mas')
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            ax[0].set_xlabel('lambda/D')
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            if pxscl2.unit==(u.meter/u.pix): ax[1].set_xlabel('meters')
            elif pxscl2.unit==(u.millimeter/u.pix): ax[1].set_xlabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[1].set_xlabel('arcsec')
            elif pxscl2.unit==(u.mas/u.pix): ax[1].set_xlabel('mas')
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            ax[1].set_xlabel('lambda/D')
    else:
        extent2=None
    
    if lognorm1: norm1 = LogNorm(vmin=vmin1,vmax=vmax1)
    else: norm1 = Normalize(vmin=vmin1,vmax=vmax1)   
    if lognorm2: norm2 = LogNorm(vmin=vmin2,vmax=vmax2)
    else: norm2 = Normalize(vmin=vmin2,vmax=vmax2)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(wspace=wspace)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def myimshow3(arr1, arr2, arr3,
              title1=None, title2=None, title3=None, titlesize=12,
              npix=None, 
              pxscl1=None, pxscl2=None, pxscl3=None, 
              use_ylabel1=True, use_ylabel2=False, use_ylabel3=False,
              cmap1='magma', cmap2='magma', cmap3='magma',
              lognorm1=False, lognorm2=False, lognorm3=False,
              vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None, 
              patches1=None, patches2=None, patches3=None,
              display_fig=True, 
              return_fig=False, 
              figsize=(10,4), dpi=125, wspace=0.25):
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    
    if cp and isinstance(arr1, cp.ndarray):
        arr1 = arr1.get()
    if cp and isinstance(arr2, cp.ndarray):
        arr2 = arr2.get()
    if cp and isinstance(arr3, cp.ndarray):
        arr3 = arr3.get()
    
    if npix is not None:
        arr1 = pad_or_crop(arr1, npix)
        arr2 = pad_or_crop(arr2, npix)
        arr3 = pad_or_crop(arr3, npix)
    
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            
        if isinstance(pxscl2, u.Quantity) and use_ylabel1:
            if pxscl2.unit==(u.meter/u.pix): ax[0].set_ylabel('meters')
            elif pxscl2.unit==(u.millimeter/u.pix): ax[0].set_ylabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[0].set_ylabel('arcsec')
            elif pxscl2.unit==(u.mas/u.pix): ax[0].set_ylabel('mas')
        elif not isinstance(pxscl2, u.Quantity) and use_ylabel1:
            ax[0].set_ylabel('lambda/D')
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
        
        if isinstance(pxscl2, u.Quantity) and use_ylabel2:
            if pxscl2.unit==(u.meter/u.pix): ax[1].set_ylabel('meters')
            elif pxscl2.unit==(u.millimeter/u.pix): ax[1].set_ylabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[1].set_ylabel('arcsec')
            elif pxscl2.unit==(u.mas/u.pix): ax[1].set_ylabel('mas')
        elif not isinstance(pxscl2, u.Quantity) and use_ylabel2:
            ax[1].set_ylabel('lambda/D')
    else:
        extent2=None
        
    if pxscl3 is not None:
        if isinstance(pxscl3, u.Quantity):
            vext = pxscl3.value * arr3.shape[0]/2
            hext = pxscl3.value * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
            
        if isinstance(pxscl2, u.Quantity) and use_ylabel3:
            if pxscl2.unit==(u.meter/u.pix): ax[2].set_ylabel('meters')
            elif pxscl2.unit==(u.millimeter/u.pix): ax[2].set_ylabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[2].set_ylabel('arcsec')
            elif pxscl2.unit==(u.mas/u.pix): ax[2].set_ylabel('mas')
        elif not isinstance(pxscl2, u.Quantity) and use_ylabel3:
            ax[2].set_ylabel('lambda/D')
    else:
        extent3=None
    
    if lognorm1: norm1 = LogNorm(vmin=vmin1,vmax=vmax1)
    else: norm1 = Normalize(vmin=vmin1,vmax=vmax1)   
    if lognorm2: norm2 = LogNorm(vmin=vmin2,vmax=vmax2)
    else: norm2 = Normalize(vmin=vmin2,vmax=vmax2)
    if lognorm3: norm3 = LogNorm(vmin=vmin3,vmax=vmax3)
    else: norm3 = Normalize(vmin=vmin3,vmax=vmax3)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    ax[0].set_xticks([])
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1, fontsize=titlesize)
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("bottom", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    ax[1].set_xticks([])
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2, fontsize=titlesize)
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("bottom", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    
    # third plot
    im = ax[2].imshow(arr3, cmap=cmap3, norm=norm3, extent=extent3, aspect='equal')
    ax[2].set_xticks([])
    ax[2].tick_params(axis='x', labelsize=9, rotation=30)
    ax[2].tick_params(axis='y', labelsize=9, rotation=30)
    if patches3: 
        for patch3 in patches3:
            ax[2].add_patch(patch3)
    ax[2].set_title(title3, fontsize=titlesize)
    
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("bottom", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    
    plt.subplots_adjust(wspace=wspace)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = cp.zeros((npix,npix), dtype=arr_in.dtype) if cupy_available and isinstance(arr_in, cp.ndarray) else np.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

# functions for saving python data
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data
        