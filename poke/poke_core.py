# get the poke submodules that get called here
from poke.poke_math import np
import poke.plotting as plot
import poke.polarization as pol
import poke.gbd as gbd
import poke.beamlets as beam
import poke.raytrace as rt


""" THE RULES
1) No physics here, all physics get their own separate module
2) Simple translation is allowed
3) No plotting/writing here, call other functions

This will be a beast of a script so I want it to be readable
"""

class Rayfront:

    def __init__(self,nrays,wavelength,pupil_radius,max_fov,normalized_pupil_radius=1,fov=[0.,0.],waist_pad=None,circle=True):


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
            if waist_pad:
                wo = waist_pad
            else:
                wo = 0
            x = x[np.sqrt(X**2 + Y**2) < self.raybundle_extent-wo/2] 
            y = y[np.sqrt(X**2 + Y**2) < self.raybundle_extent-wo/2]

        x = np.ravel(x)/pupil_radius
        y = np.ravel(y)/pupil_radius

        print('norm fov = ',self.normalized_fov)

        # in normalized pupil and field coords for an on-axis field
        self.base_rays = np.array([x,
                                   y,
                                   0*x + self.normalized_fov[0],
                                   0*y + self.normalized_fov[1]])
        print('base ray shape ',self.base_rays.shape)
    # First optional constructors of our core physics modules

    #@classmethod
    def as_gaussianbeamlets(self,wo):

        """optional constructor to init the rayfront for GBD, comes with additional args

        Parameters
        ----------
        wo : float
            The gaussian beam waist used to decompose the field. Coupled to nrays and OF
        """

        # gaussian beam parameters
        self.wo = wo
        self.div = self.wavelength/(np.pi*self.wo) * 180 / np.pi # beam divergence in deg

        # ray differentials in normalized coords
        dPx = self.wo/self.pupil_radius
        dPy = self.wo/self.pupil_radius
        dHx = self.div/self.max_fov
        dHy = self.div/self.max_fov

        # differential ray bundles from base rays
        self.Px_rays = np.copy(self.base_rays)

        if np.__name__ == 'jax.numpy':
            self.Px_rays = self.Px_rays.at[0].set(dPx)

            self.Py_rays = np.copy(self.base_rays)
            self.Py_rays = self.Px_rays.at[1].set(dPy)

            self.Hx_rays = np.copy(self.base_rays)
            self.Hx_rays = self.Hx_rays.at[2].set(dHx)

            self.Hy_rays = np.copy(self.base_rays)
            self.Hy_rays = self.Hy_rays.at[3].set(dHy)

        else:
            self.Px_rays[0] += dPx

            self.Py_rays = np.copy(self.base_rays)
            self.Py_rays[1] += dPy

            self.Hx_rays = np.copy(self.base_rays)
            self.Hx_rays[2] += dHx

            self.Hy_rays = np.copy(self.base_rays)
            self.Hy_rays[3] += dHy

        # total set of rays
        self.raysets = [self.base_rays,self.Px_rays,self.Py_rays,self.Hx_rays,self.Hy_rays]

        # Will force the transverse coords to be x and y
        self.global_coords = False

        # We want to save the differential quantities
        self.dPx = dPx
        self.dPy = dPy
        self.dHx = dHx
        self.dHy = dHy

    
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

        self._surfaces = surfaces # a list of dictionaries
        self.raysets = [self.base_rays]
        self.global_coords = True
        self.P_total = []
        self.jones_pupil = []

    """
    ########################### GENERAL RAY TRACING METHODS ###########################
    """

    def trace_rayset(self,pth,wave=1,surfaces=None):

        if surfaces != None:
            self._surfaces = surfaces

        if (pth[-3:] == 'zmx') or (pth[-3:] == 'zos'):
            positions,directions,normals,self.opd = rt.TraceThroughZOS(self.raysets,pth,self._surfaces,self.nrays,wave,self.global_coords)
        elif (pth[-3:] == 'seq') or (pth[-3:] == 'len'):
            positions,directions,normals,self.opd = rt.TraceThroughCV(self.raysets,pth,self._surfaces,self.nrays,wave,self.global_coords)

        self.xData = positions[0]
        self.yData = positions[1]
        self.zData = positions[2]

        self.lData = directions[0]
        self.mData = directions[1]
        self.nData = directions[2]
        
        # Keep sign in raytracer coordinate system
        self.l2Data = normals[0]
        self.m2Data = normals[1]
        self.n2Data = normals[2]


    def TraceRaysetZOS(self,pth,wave=1,surfaces=None):

        print('this function is depreciated, please use trace_rayset')
        if surfaces != None:
            self._surfaces = surfaces

        """Traces rays through zemax opticstudio

        xData (etc.) has shape [len(raysets),len(surflist),maxrays] from TraceThroughZOS
        """

        positions,directions,normals,self.opd = rt.TraceThroughZOS(self.raysets,pth,self._surfaces,self.nrays,wave,self.global_coords)
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

        # We should update the raysets! What's the best way to do this ...

    def TraceRaysetCV(self,pth,wave=1,surfaces=None):
        
        print('this function is depreciated, please use trace_rayset')
        if surfaces != None:
            self._surfaces = surfaces

        positions,directions,normals,self.opd = rt.TraceThroughCV(self.raysets,pth,self._surfaces,self.nrays,wave,self.global_coords)
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
    def beamlet_decomposition_field(self,dcoords,dnorms=np.array([0.,0.,1.]),memory_avail=4,misaligned=True):
        """computes the coherent field by decomposing the entrance pupil into gaussian beams
        and propagating them to the final surface

        Parameters
        ----------
        dcoords : Nx3 numpy.ndarray
            coordinates of detector pixels
        dnorms : Nx3 numpy.ndarray
            coordinates of detector pixel surface normals. Nominally useful for tilted or curved detectors. 
            Defaults to pointing along the local z-axis of the detector surface
        memory_avail : int
            amount of memory in GB to use for field calculation
        """

        # converting memory
        nrays = self.nData[:,-1].shape[1]
        npix = dcoords.shape[-1] # need to have coords in first dimension and be raveled
        print('pixels = ',npix)
        total_size = nrays*npix*128*4 * 1e-9 # complex128, 4 is a fudge factor to account for intermediate variables
        nloops = int(total_size/memory_avail)
        if nloops < 1:
            nloops = 1
        print(f'beamlet field at wavelength = {self.wavelength}')

        if misaligned:
            field = beam.misaligned_beamlet_field(self.xData,self.yData,self.zData,self.lData,self.mData,self.nData,self.opd,
                                                    self.wo,self.wo,self.div*np.pi/180,self.div*np.pi/180, dcoords,dnorms,
                                                    wavelength=self.wavelength,nloops=nloops,use_centroid=True)
        else:

            field = beam.beamlet_decomposition_field(self.xData,self.yData,self.zData,self.lData,self.mData,self.nData,self.opd,
                                                    self.wo,self.wo,self.div*np.pi/180,self.div*np.pi/180, dcoords,dnorms,
                                                    wavelength=self.wavelength,nloops=nloops,use_centroid=True)
            
        return field



    def EvaluateGaussianField(self,detsize,npix,return_cube=False):

        print('this function is depreciated, please use beamlet_decomposition_field()')

        """Computes the coherent field as a finite sum of gaussian beams

        Parameters
        ----------
        detsize : float
            full side length of a square detector centered on the optical axis of the final surface

        return_cube : bool
            whether to return the rays coherently summed or separated. Setting to True makes nice movies.
            optional, defaults to False

        Returns
        -------

        gaussfield : complex128 ndarray
            array containing the complex field of the gaussian beamlets

        """

        gaussfield = gbd.EvalField(self.xData,self.yData,self.zData,self.lData,self.mData,self.nData,self.opd,
                                   self.wo,self.wo,self.div*np.pi/180,self.div*np.pi/180,detsize,npix)

        if return_cube == False:

            if np.__name__ == "cupy":

                gaussfield = gaussfield.reshape([npix,npix]).get()

            else:

                gaussfield = gaussfield.reshape([npix,npix])

        return gaussfield

    
    """ 
    ########################### POLARIZATION RAY TRACING METHODS ###########################
    """

    def compute_jones_pupil(self,ambient_index=1,aloc=np.array([0.,0.,1.]),exit_x=np.array([1.,0.,0.]),proper_retardance=False):

        for rayset_ind,rayset in enumerate(self.raysets):


            aoi,kin,kout,norm = rt.ConvertRayDataToPRTData(self.lData[rayset_ind],self.mData[rayset_ind],self.nData[rayset_ind],
                                                            self.l2Data[rayset_ind],self.m2Data[rayset_ind],self.n2Data[rayset_ind],
                                                            self._surfaces)
            
            Psys,Jsys,Qsys = pol.system_prt_matrices(aoi,kin,kout,norm,self._surfaces,self.wavelength,ambient_index)
            P,Q = pol.total_prt_matrix(Psys,Qsys)
            if proper_retardance:
                Jpupil = pol.global_to_local_coordinates(P,kin[0],kout[-1],aloc,exit_x,Q=Q)
            else:
                Jpupil = pol.global_to_local_coordinates(P,kin[0],kout[-1],aloc,exit_x)

            self.jones_pupil.append(Jpupil)
            self.P_total.append(P)
            


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
                                                            self._surfaces)


            # Hold onto J and O for now
            # we are just gonna use P
            P,J = rt.ComputePRTMatrixFromRayData(aoi,kin,kout,norm,self._surfaces,self.wavelength,ambient_index)
            self.P_total.append(rt.ComputeTotalPRTMatrix(P))
            self.JonesPupil.append(rt.PRTtoJonesMatrix(self.P_total[rayset_ind],kin[0],kout[-1],aloc,exit_x))
    
    def ComputeARM(self,pad=2,circle=True):
        """Computes the amplitude response matrix from the Jones Pupil, requires a square array
        """
        
        J = self.JonesPupil[-1][:,:2,:2]
        J_dim = int(np.sqrt(J.shape[0]))
        
        A = np.empty([J_dim*pad,J_dim*pad,2,2],dtype='complex128')
        J = np.reshape(J,[J_dim,J_dim,2,2])
        
        # Create a circular aperture
        x = np.linspace(-1,1,J.shape[0])
        x,y = np.meshgrid(x,x)
        mask = np.ones([J.shape[0],J.shape[0]])
        
        if circle:
            mask[x**2 + y**2 > 1] = 0

        for i in range(2):
            for j in range(2):
                A[...,i,j] = np.fft.fftshift(np.fft.fft2(np.pad(J[...,i,j]*mask,int(J_dim*pad/2-(J_dim/2)))))
                
        self.ARM = A
        return A
        
    def ComputePSM(self,cut=128,stokes=np.array([1.,0.,0.,0.])):
    
        """
        We regrettably need to loop over this because we use numpy.kron()
        """
        
        # cut out the center
        size = self.ARM.shape[0]/2
        A = self.ARM[int(size-cut):int(size+cut),int(size-cut):int(size+cut)]
        P = np.empty([A.shape[0],A.shape[1],4,4])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
            
                P[i,j] = pol.JonesToMueller(A[i,j])
                
        img = P @ stokes
        self.PSM = P
        return img[...,0]        
                
    """ 
    ########################### PROPERTIES ###########################
    """

    @property
    def surfaces(self):
        return self._surfaces
    
    @surfaces.setter
    def surfaces(self,surflist):
        self._surfaces = surflist


            
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

    """ 
    ########################### Source Module Conversions ###########################
    """

    def convert_data_sourcemodule(self,new_backend='numpy'):
        """This is a bit cursed, but in the case where data is initialized in numpy, but we want to use it in Jax/Cupy, then we have to convert it
        """

        from poke.poke_math import np,set_backend_to_cupy,set_backend_to_jax,set_backend_to_numpy # make sure we have the current source module loaded

        if new_backend == 'numpy':

            set_backend_to_numpy()

        elif new_backend == 'jax':
            
            set_backend_to_jax()

        elif new_backend == 'cupy':

            set_backend_to_cupy()

        else:
            print('Did not recognize module, defaulting to numpy')
            set_backend_to_numpy()


        # Ray Data
        self.xData = np.asarray(self.xData)
        self.yData = np.asarray(self.yData)
        self.zData = np.asarray(self.zData)

        self.lData = np.asarray(self.lData)
        self.mData = np.asarray(self.mData)
        self.nData = np.asarray(self.nData)

        self.l2Data = np.asarray(self.l2Data)
        self.m2Data = np.asarray(self.m2Data)
        self.n2Data = np.asarray(self.n2Data)

        self.opd = np.asarray(self.opd)