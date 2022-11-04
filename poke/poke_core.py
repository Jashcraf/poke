# poke_core.py
import numpy as np
import poke.poke_math as mat
import poke.writing as write
import poke.thinfilms_prysm as tf
import poke.plotting as plot
import poke.raytrace as rt
import poke.polarization as pol

# Make optional
import zosapi

""" THE RULES
1) No physics here, all physics get their own separate module
2) Simple translation is allowed
3) No plotting/writing here, call other functions

This will be a beast of a script so I want it to be readable
"""

class Rayfront:

    def __init__(self,nrays,wavelength,pupil_radius,max_fov,normalized_pupil_radius=1,fov=[0.,0.],circle=True):

        """class for the Rayfront object that 
        1) traces rays with the zosapi
        2) uses ray data to do GBD
        3) uses ray data to do PRT

        Parameters
        ----------
        nrays : int
            number of rays across a square pupil

        wavelength : float
            wavelength of light in meters

        pupil_radius : float
            radius of the entrance pupil in meters

        max_fov : float
            radius of the field of view in degrees

        normalized_pupil_radius : float, optional
            fraction of the pupil radius to actually trace, defaults to 1

        fov : tuple of floats, optional
            fov of the raybundle to trace in x (fov[0]) and y (fov[1]), 
            like a shift in the angular dimension, defaults to 0

        circe : bool, optional
            whether to crop the square aperture to a circle of rays, defaults to True

        """

        self.nrays = nrays # rays across a square pupil
        self.wavelength = wavelength
        self.pupil_radius = pupil_radius
        self.normalized_pupil_radius = normalized_pupil_radius # normalized radius
        self.max_fov = max_fov
        self.fov = np.array(fov)

        self.normalized_fov = self.fov/max_fov
        self.raybundle_extent = pupil_radius*normalized_pupil_radius # the actual extent of the raybundle

        # init rayset
        # init raysets
        x = np.linspace(-self.raybundle_extent,self.raybundle_extent,nrays)
        x,y = np.meshgrid(x,x)
        X = x
        Y = y
        
        if circle == True:

            x = x[np.sqrt(X**2 + Y**2) < self.raybundle_extent] 
            y = y[np.sqrt(X**2 + Y**2) < self.raybundle_extent]

        x = np.ravel(x)/pupil_radius
        y = np.ravel(y)/pupil_radius

        # in normalized pupil and field coords for an on-axis field
        self.base_rays = np.array([x,
                                   y,
                                   0*x + self.normalized_fov[0],
                                   0*y + self.normalized_fov[1]])
    # First optional constructors of our core physics modules

    #@classmethod
    def as_gaussianbeamlets(cls,wo):

        """optional constructor to init the rayfront for GBD, comes with additional args

        Parameters
        ----------
        wo : float
            The gaussian beam waist used to decompose the field. Coupled to nrays and OF
        """

        import poke.gbd as gbd

        # gaussian beam parameters
        cls.wo = wo
        cls.div = cls.wavelength/(np.pi*cls.wo) # beam divergence

        # ray differentials in normalized coords
        dPx = wo/cls.pupil_radius
        dPy = wo/cls.pupil_radius
        dHx = cls.div/cls.max_fov
        dHy = cls.div/cls.max_fov

        # differential ray bundles from base rays
        cls.Px_rays = cls.base_rays
        cls.Px_rays[0] += dPx

        cls.Py_rays = cls.base_rays
        cls.Py_rays[1] += dPy

        cls.Hx_rays = cls.base_rays
        cls.Hx_rays[2] += dHx

        cls.Hy_rays = cls.base_rays
        cls.Hy_rays[3] += dHy

        # total set of rays
        cls.raysets = [cls.base_rays,cls.Px_rays,cls.Py_rays,cls.Hx_rays,cls.Hy_rays]

    
    #@classmethod
    def as_polarized(self,surfaces):

        """optional constructor to init the rayfront for PRT, comes with additional args

        Parameters
        ----------
        surfaces : list of dicts
            list of dictionaries that contain {
                'surf' : surface number in raytrace code
                'coating' : complex float or list
                'mode' : 'reflect' or 'refract'
                }
        """

        # surfaces is a list of dictionaries

        # system dicts are surf
        import poke.polarization as pol

        self.surfaces = surfaces # a list of dictionaries
        self.raysets = [self.base_rays]

    """
    ########################### GENERAL RAY TRACING METHODS ###########################
    """

    def TraceRaysetZOS(self,pth,wave=1,global_coords=True):

        """Traces rays through zemax opticstudio
        """


        positions,directions,normals,self.opd = rt.TraceThroughZOS(self.raysets,pth,self.surfaces,self.nrays,wave,global_coords)
        # Remember that these dimensions are
        # 0 : rayset
        # 1 : surface #
        # 2 : ray coordinate value

        self.xData = positions[0]
        self.yData = positions[1]
        self.zData = positions[2]

        self.lData = directions[0]
        self.mData = directions[1]
        self.nData = directions[2]
        
        # Keep sign in zmx coordinate system
        self.l2Data = normals[0]
        self.m2Data = normals[1]
        self.n2Data = normals[2]
    
    """ 
    ########################### GAUSSIAN BEAMLET TRACING METHODS ###########################
    """

    def EvaluateGaussianField(self):
        pass

    
    """ 
    ########################### POLARIZATION RAY TRACING METHODS ###########################
    """

    def ComputeJonesPupil(self,ambient_index=1,aloc=np.array([0.,0.,1.]),exit_x=np.array([1.,0.,0.])):

        """Computes the Jones Pupil, PRT Matrix, and Parallel Transport
        """

        # Init the values to compute
        self.P_total = []
        self.JonesPupil = []

        # Loop over raysets
        for rayset_ind,rayset in enumerate(self.raysets):

            # These outputs are lists, where each element is the data at a given surface for all rays
            # I think these objects are actually accessing the data per surface, rather than the rayset
            aoi,kin,kout,norm= rt.ConvertRayDataToPRTData(self.lData[rayset_ind],self.mData[rayset_ind],self.nData[rayset_ind],
                                                            self.l2Data[rayset_ind],self.m2Data[rayset_ind],self.n2Data[rayset_ind],
                                                            self.surfaces)

            # Plot kin
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            # fig,ax = plt.subplots(ncols=len(kin))
            # for i,axs in enumerate(ax):
                
            #     axs.set_title('Surface {}'.format(i+1))
            #     im = axs.scatter(self.xData[0,i],self.yData[0,i],c=aoi[i])
            #     fig.colorbar(im)
            # plt.show()

            # fig,ax = plt.subplots(ncols=len(kin))
            # for i,axs in enumerate(ax):
                # rs,rp = pol.FresnelCoefficients(aoi[i],1,self.surfaces[1]['coating'])
                # axs.set_title('Surface {} rs'.format(i+1))
                # im = axs.scatter(self.xData[0,i],self.yData[0,i],c=np.angle(rs))
                # divider = make_axes_locatable(axs)
                # cax = divider.append_axes("right",size="5%",pad="2%")
                # fig.colorbar(im,cax=cax)
            # plt.show()
            
            # fig,ax = plt.subplots(ncols=len(kin))
            # for i,axs in enumerate(ax):
                # rs,rp = pol.FresnelCoefficients(aoi[i],1,self.surfaces[1]['coating'])
                # axs.set_title('Surface {} rp'.format(i+1))
                # im = axs.scatter(self.xData[0,i],self.yData[0,i],c=np.angle(rp))
                # divider = make_axes_locatable(axs)
                # cax = divider.append_axes("right",size="5%",pad="2%")
                # fig.colorbar(im,cax=cax)
            # plt.show()


            # Hold onto J and O for now
            # we are just gonna use P
            P,J = rt.ComputePRTMatrixFromRayData(aoi,kin,kout,norm,self.surfaces,self.wavelength,ambient_index)
            self.P_total.append(rt.ComputeTotalPRTMatrix(P))
            self.JonesPupil.append(rt.PRTtoJonesMatrix(self.P_total[rayset_ind],kin[0],kout[-1],aloc,exit_x))

    """ 
    ########################### DATA PLOTTING/VISUALIZE METHODS ###########################
    """

    def PlotGaussianField(self):

        pass

    def PlotRaysAtSurface(self,surf,rayset_number=0):

        plot.PlotRayset(rayset_number,self.xData,self.yData,self.lData,self.mData,surf=surf)

    def PlotJonesPupil(self,rayset_ind=0):

        plot.PlotJonesPupil(self.xData[rayset_ind,0],self.yData[rayset_ind,0],self.JonesPupil[rayset_ind])

    def PlotPRTMatrix(self,rayset_ind=0):

        plot.PlotJonesPupil(self.xData[rayset_ind,-1],self.yData[rayset_ind,-1],self.P_total[rayset_ind])


    """ 
    ########################### DATA WRITE TO TEXT/FITS METHODS ###########################
    """

    def WriteFieldToFITS(self):
        pass

    def WritePRTMatrixToFits(self):
        pass

    def WriteJonesPupilToFits(self):
        pass