import warnings

# get the poke submodules that get called here
from poke.poke_math import np
import poke.plotting as plot
import poke.polarization as pol

# import poke.gbd as gbd
import poke.beamlets as beam
import poke.raytrace as rt
import poke.interfaces as inter


""" THE RULES
1) No physics here, all physics get their own separate module
2) Simple translation is allowed
3) No plotting/writing here, call other functions
"""

GOLDEN = (1 + np.sqrt(5)) / 2


class Rayfront:
    def __init__(self, nrays, wavelength, pupil_radius, max_fov, normalized_pupil_radius=1, fov=[0.0, 0.0], waist_pad=None, circle=True, grid="even",):

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

        self.nrays = nrays  # rays across a square pupil
        self.wavelength = wavelength
        self.pupil_radius = pupil_radius
        self.normalized_pupil_radius = normalized_pupil_radius  # normalized radius
        self.max_fov = max_fov
        self.fov = np.array(fov)

        self.normalized_fov = self.fov / max_fov
        self.raybundle_extent = (
            pupil_radius * normalized_pupil_radius
        )  # the actual extent of the raybundle

        # init rayset
        # init raysets
        x = np.linspace(-self.raybundle_extent, self.raybundle_extent, nrays)
        x, y = np.meshgrid(x, x)
        X = x
        Y = y

        if circle == True:
            if waist_pad:
                wo = waist_pad
            else:
                wo = 0
            x = x[np.sqrt(X ** 2 + Y ** 2) < self.raybundle_extent - wo / 4]
            y = y[np.sqrt(X ** 2 + Y ** 2) < self.raybundle_extent - wo / 4]

            if grid == "fib":
                i = len(x)  # use however many rays are in a circular aperture with even sampling
                n = np.arange(1, i)
                Rn = np.sqrt(n / i)
                Tn = 2 * np.pi / GOLDEN ** 2 * n
                x_fib = Rn * np.cos(Tn)
                y_fib = Rn * np.sin(Tn)
                x = x_fib
                y = y_fib

        x = np.ravel(x) / pupil_radius
        y = np.ravel(y) / pupil_radius

        print("norm fov = ", self.normalized_fov)

        # in normalized pupil and field coords for an on-axis field
        self.base_rays = np.array(
            [x, y, 0 * x + self.normalized_fov[0], 0 * y + self.normalized_fov[1]]
        )
        print("base ray shape ", self.base_rays.shape)

    # First optional constructors of our core physics modules

    # @classmethod
    def as_gaussianbeamlets(self, wo):

        """optional constructor to init the rayfront for GBD, comes with additional args

        Parameters
        ----------
        wo : float
            The gaussian beam waist used to decompose the field. Coupled to nrays and OF
        """

        # gaussian beam parameters
        self.wo = wo
        self.div = self.wavelength / (np.pi * self.wo) * 180 / np.pi  # beam divergence in deg

        # ray differentials in normalized coords
        dPx = self.wo / self.pupil_radius
        dPy = self.wo / self.pupil_radius
        dHx = self.div / self.max_fov
        dHy = self.div / self.max_fov

        # differential ray bundles from base rays
        self.Px_rays = np.copy(self.base_rays)

        if np.__name__ == "jax.numpy":
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
        self.raysets = [self.base_rays, self.Px_rays, self.Py_rays, self.Hx_rays, self.Hy_rays]

        # Will force the transverse coords to be x and y
        self.global_coords = False

        # We want to save the differential quantities
        self.dPx = dPx
        self.dPy = dPy
        self.dHx = dHx
        self.dHy = dHy

    # @classmethod
    def as_polarized(self, surfaces):

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

        self._surfaces = surfaces  # a list of dictionaries
        self.raysets = [self.base_rays]
        self.global_coords = True
        self.P_total = []
        self.jones_pupil = []

    """
    ########################### GENERAL RAY TRACING METHODS ###########################
    """

    def trace_rayset(self, pth, wave=1, surfaces=None, ref_surf=1, _experimental=True):
        """
        Parameters
        ----------
        pth : str
            path to the lens file you want to run the ray trace on
        wave : int
            wavelength number in the lens data editor to run ray trace for, Defaults to 1
        surfaces : list of dictionaries
            List of surface dictionaries to guide the raytrace. Optional, if None uses self.surflist
        ref_surf : int
            Global coordinate reference surface, defaults to 1
        _experimental : boolean
            If True, default to the faster code v ray trace
        """
        if surfaces != None:
            self._surfaces = surfaces

        if (pth[-3:] == "zmx") or (pth[-3:] == "zos"):
            positions, directions, normals, self.opd, self.vignetted = rt.trace_through_zos(
                self.raysets, pth, self._surfaces, self.nrays, wave, self.global_coords
            )
        elif (pth[-3:] == "seq") or (pth[-3:] == "len"):
            if _experimental:
                positions, directions, normals, self.opd = rt.trace_through_cv(
                    self.raysets, pth, self._surfaces, self.nrays, wave, self.global_coords, 
                    global_coord_reference=ref_surf)
            else:
                positions, directions, normals, self.opd = rt.TraceThroughCV(
                    self.raysets, pth, self._surfaces, self.nrays, wave, self.global_coords
                )

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

    """ 
    ########################### GAUSSIAN BEAMLET TRACING METHODS ###########################
    """

    def beamlet_decomposition_field(self, dcoords, dnorms=np.array([0.0, 0.0, 1.0]), memory_avail=4, misaligned=True, vignette=True):
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
        nrays = self.nData[:, -1].shape[1]
        npix = dcoords.shape[-1]  # need to have coords in first dimension and be raveled
        print("pixels = ", npix)
        print("rays = ", nrays)
        total_size = (
            nrays * npix * 128 * 1e-9
        )  # complex128, 4 is a fudge factor to account for intermediate variables
        total_blocks = total_size / memory_avail
        nloops = np.ceil(total_blocks)
        if nloops < 1:
            nloops = 1
        print(f"beamlet field at wavelength = {self.wavelength}")

        if misaligned:
            if vignette:
                field = beam.misaligned_beamlet_field(
                    self.xData,
                    self.yData,
                    self.zData,
                    self.lData,
                    self.mData,
                    self.nData,
                    self.opd,
                    self.wo,
                    self.wo,
                    self.div * np.pi / 180,
                    self.div * np.pi / 180,
                    dcoords,
                    dnorms,
                    wavelength=self.wavelength,
                    nloops=nloops,
                    use_centroid=True,
                    vignetting=self.vignetted,
                )
            else:
                field = beam.misaligned_beamlet_field(
                    self.xData,
                    self.yData,
                    self.zData,
                    self.lData,
                    self.mData,
                    self.nData,
                    self.opd,
                    self.wo,
                    self.wo,
                    self.div * np.pi / 180,
                    self.div * np.pi / 180,
                    dcoords,
                    dnorms,
                    wavelength=self.wavelength,
                    nloops=nloops,
                    use_centroid=True,
                )
        else:

            field = beam.beamlet_decomposition_field(
                self.xData,
                self.yData,
                self.zData,
                self.lData,
                self.mData,
                self.nData,
                self.opd,
                self.wo,
                self.wo,
                self.div * np.pi / 180,
                self.div * np.pi / 180,
                dcoords,
                dnorms,
                wavelength=self.wavelength,
                nloops=int(nloops),
                use_centroid=True,
                vignetting=self.vignetted,
            )

        return field

    """ 
    ########################### POLARIZATION RAY TRACING METHODS ###########################
    """

    def compute_jones_pupil(
        self,
        ambient_index=1,
        aloc=np.array([0.0, 0.0, 1.0]),
        entrance_x=np.array([1.0, 0.0, 0.0]),
        exit_x=np.array([1.0, 0.0, 0.0]),
        proper_retardance=False,
        coordinates="double",
        collimated_object=True):
        """compute jones pupil from ray data using the double pole coordinate system

        Parameters
        ----------
        ambient_index : float, optional
            refractive index the system is immersed in, by default 1
        aloc : numpy.ndarray, optional
            direction of the double antipole - typically use the optical axis ray direction cosines, by default np.array([0.,0.,1.])
        entrance_x : numpy.ndarray, optional
            input local x-axis in global coordinates, by default np.array([1.,0.,0.])
        exit_x : numpy.ndarray, optional
            output local x-axis in global coordinates, by default np.array([1.,0.,0.])
        proper_retardance : bool, optional
            whether to use the "proper" retardance calculation, by default False
        coordinates : string
            type of local coordinate transformation to use. Options are "double" for the double-pole
            coordinate system, and "dipole" for the dipole coordinate system.
        collimated_object : bool
            If object space is collimated or not. If true, applies same basis vectors to all input rays,
            otherwise, it computes them on a curved surface.
        """

        if proper_retardance:
            warnings.warn("The proper retardance calculation is prone to unphysical results and requires further testing")

        for rayset_ind, rayset in enumerate(self.raysets):

            aoi, kin, kout, norm = rt.convert_ray_data_to_prt_data(
                self.lData[rayset_ind],
                self.mData[rayset_ind],
                self.nData[rayset_ind],
                self.l2Data[rayset_ind],
                self.m2Data[rayset_ind],
                self.n2Data[rayset_ind],
                self._surfaces,
                ambient_index=ambient_index
            )

            Psys, Jsys, Qsys = pol.system_prt_matrices(aoi, kin, kout, norm, self._surfaces, self.wavelength, ambient_index)
            P, Q = pol.total_prt_matrix(Psys, Qsys)

            if proper_retardance:
                Jpupil = pol.global_to_local_coordinates(P, kin[0], kout[-1], aloc, entrance_x, exit_x, Q=Q, coordinates=coordinates)

            else:
                Jpupil = pol.global_to_local_coordinates(P, kin[0], kout[-1], aloc, entrance_x, exit_x, coordinates=coordinates)

            self.jones_pupil.append(Jpupil)
            self.P_total.append(P)

    def compute_arm(self, pad=2, circle=True, is_square=True):
        """Computes the amplitude response matrix from the Jones Pupil, requires a square array
        """

        if is_square:

            J = self.JonesPupil[-1][:, :2, :2]
            J_dim = int(np.sqrt(J.shape[0]))
            J = np.reshape(J, [J_dim, J_dim, 2, 2])

        else:

            # Expand into a polynomial basis
            J = inter.regularly_space_jones(self, 11, self.nrays)

        A = np.empty([J_dim * pad, J_dim * pad, 2, 2], dtype="complex128")

        # Create a circular aperture
        x = np.linspace(-1, 1, J.shape[0])
        x, y = np.meshgrid(x, x)
        mask = np.ones([J.shape[0], J.shape[0]])

        if circle:
            mask[x ** 2 + y ** 2 > 1] = 0

        for i in range(2):
            for j in range(2):
                A[..., i, j] = np.fft.fftshift(
                    np.fft.fft2(np.pad(J[..., i, j] * mask, int(J_dim * pad / 2 - (J_dim / 2))))
                )

        self.ARM = A
        return A

    def compute_psm(self, cut=128, stokes=np.array([1.0, 0.0, 0.0, 0.0])):

        """
        We regrettably need to loop over this because we use numpy.kron()
        """

        # cut out the center
        size = self.ARM.shape[0] / 2
        A = self.ARM[int(size - cut) : int(size + cut), int(size - cut) : int(size + cut)]
        P = np.empty([A.shape[0], A.shape[1], 4, 4])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):

                P[i, j] = pol.jones_to_mueller(A[i, j])

        img = P @ stokes
        self.PSM = P
        return img[..., 0]

    """ 
    ########################### PROPERTIES ###########################
    """

    @property
    def surfaces(self):
        return self._surfaces

    @surfaces.setter
    def surfaces(self, surflist):
        self._surfaces = surflist

    """ 
    ########################### Source Module Conversions ###########################
    """

    def convert_data_sourcemodule(self, new_backend="numpy"):
        """This is a bit cursed, but in the case where data is initialized in numpy, but we want to use it in Jax/Cupy, then we have to convert it
        and vice versa
        """

        from poke.poke_math import (
            np,
            set_backend_to_cupy,
            set_backend_to_jax,
            set_backend_to_numpy,
        )  # make sure we have the current source module loaded

        if new_backend == "numpy":

            set_backend_to_numpy()

        elif new_backend == "jax":

            set_backend_to_jax()

        elif new_backend == "cupy":

            set_backend_to_cupy()

        else:
            print("Did not recognize module, defaulting to numpy")
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
