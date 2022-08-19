import numpy as np
import zosapi
import poke.poke_core as pol
import poke.poke_math as mat
import poke.writing as write
from poke.gbd import * 
import poke.thinfilms_prysm as tf

class Rayfront:

    def __init__(self,nrays,n1,n2,wavelength,mode='reflection',dPx=0,dPy=0,dHx=0,dHy=0,circle=False,stack=None):
        """Init function for raybundle class. Holds all raydata and information obtained from ray data

        Parameters
        ----------
        nrays : int
            number of rays across an aperture
        n1 : float
            Index of Refraction of medium system is in
        n2 : float
            Index of Refraction of material 
            THIS SHOULD BE UPGRADED TO ARRAY OF INDICES
        mode : str, optional
            reflection or refraction - matters for polarization ray tracing, by default 'reflection'
        dPx : int, optional
            normalized pupil coordinate differential, by default 0
        dPy : int, optional
            normalized pupil coordinate differential, by default 0
        dHx : int, optional
            normalized field coordinate differential, by default 0
        dHy : int, optional
            normalized field coordinate differential, by default 0
        circle : bool, optional
            Option to vignette rays to a circle. This looses data so it is by default False
        """

        # number of rays across a grid
        self.nrays = nrays

        # Want to add support for accepting lists of these items, for now they are singular
        self.n1 = n1
        self.n2 = n2
        self.mode = mode
        self.stack = stack # None or a prysm-acceptible stack 
        self.wavelength = wavelength

        # Add alternative constructors as class method instead of shimming in the beam waist
        # rfrnt.as_gaussfield
        # rfront.as_prtfield etc.
        wo = 0*.04/2.4

        # NormUnPol ray coordinates
        x = np.linspace(-1+wo,1-wo,nrays)
        x,y = np.meshgrid(x,x)
        X = np.ravel(x)
        Y = np.ravel(y)

        # Opticstudio has to pre-allocate space for a generally square ray grid
        # This is fine, except it leaves a bunch of zero values at the end
        # 
        if circle == True:
            self.Px = np.ravel(x)[np.sqrt(X**2 + Y**2)<=1.0] + dPx# 
            self.Py = np.ravel(y)[np.sqrt(X**2 + Y**2)<=1.0] + dPy#
        else:
            self.Px = np.ravel(x)+ dPx# 
            self.Py = np.ravel(y)+ dPy#
        
        print('max Px = ',np.max(self.Px))
        print('max Py = ',np.max(self.Py))
        print('min Px = ',np.min(self.Px))
        print('min Py = ',np.min(self.Py))

        # self.Px = np.ravel(x) + dPx# 
        # self.Py = np.ravel(y) + dPy#

        self.Hx = np.zeros(self.Px.shape) + dHx
        self.Hy = np.zeros(self.Py.shape) + dHy
        
        print('max Hx = ',np.max(self.Hx))
        print('max Hy = ',np.max(self.Hy))
        print('min Hx = ',np.min(self.Hx))
        print('min Hy = ',np.min(self.Hy))

    def TraceThroughZOS(self,pth,surflist,wave=1,global_coords=True):

        """Traces initialized rays through a zemax opticstudio file

        Parameters
        ----------
        pth : str
            path to Zemax opticstudio file. Supports .zmx extension, .zos is untested but should work

        surflist : list of ints
            list of surface numbers to trace to and record the position of. The rays will hit every surface in the optical system,
            this just tells the Raybundle if the information at that point should be saved

        wave : int, optional
            wavelength number in ZOS file, by default 1

        global_coords : bool, optional
            whether to use global coordinates or local coordinates. Defaults to global coordinates.
            PRT uses global coordinates
            GBD uses local coordinates

        """

        self.surflist = surflist
        print('surflist added to attributes')

        from System import Enum,Int32,Double,Array
        import clr,os
        dll = os.path.join(os.path.dirname(os.path.realpath(__file__)),r'Raytrace.dll')
        clr.AddReference(dll)

        self.xData = []
        self.yData = []
        self.zData = []

        self.lData = []
        self.mData = []
        self.nData = []

        self.l2Data = []
        self.m2Data = []
        self.n2Data = []

        self.opd = []

        # The global rotation matrix
        self.R = []

        # The global offset vector
        self.O = []

        import BatchRayTrace

        zos = zosapi.App()
        TheSystem = zos.TheSystem
        ZOSAPI = zos.ZOSAPI
        TheSystem.LoadFile(pth,False)

        if TheSystem.LDE.NumberOfSurfaces < 4:
            print('File was not loaded correctly')
            exit()
        
        if surflist[-1] > TheSystem.LDE.NumberOfSurfaces:
            print('last surface > num surfaces, setting to num surfaces')

        maxrays = self.Px.shape[0]

        for surf in surflist:

            print(surf)

            tool = TheSystem.Tools.OpenBatchRayTrace()
            normUnpol = tool.CreateNormUnpol(maxrays, ZOSAPI.Tools.RayTrace.RaysType.Real, surf)
            reader = BatchRayTrace.ReadNormUnpolData(tool, normUnpol)
            reader.ClearData()
            rays = reader.InitializeOutput(self.nrays)

            reader.AddRay(wave,self.Hx,self.Hy,self.Px,self.Py,
                          Enum.Parse(ZOSAPI.Tools.RayTrace.OPDMode,'None'))

            isfinished = False
            while not isfinished:
                segments = reader.ReadNextBlock(rays)
                if segments == 0:
                    isfinished = True

            # Global Coordinate Conversion
            sysDbl = Double(1.0)
            success,R11,R12,R13,R21,R22,R23,R31,R32,R33,XO,YO,ZO = TheSystem.LDE.GetGlobalMatrix(int(surf),
                                                                                                 sysDbl,sysDbl,sysDbl,
                                                                                                 sysDbl,sysDbl,sysDbl,
                                                                                                 sysDbl,sysDbl,sysDbl,
                                                                                                 sysDbl,sysDbl,sysDbl)

            if success != 1:
                print('Ray Failure')

            Rmat = np.array([[R11,R12,R13],
                             [R21,R22,R23],
                             [R31,R32,R33]])

            position = np.array([np.array(list(rays.X)),
                                 np.array(list(rays.Y)),
                                 np.array(list(rays.Z))])

            # I think this is just per-surface so it doesn't really need to be a big list, just a single surface.
            # Change later when cleaning up the code
            offset = np.zeros(position.shape)
            offset[0,:] = XO
            offset[1,:] = YO
            offset[2,:] = ZO

            angle = np.array([np.array(list(rays.L)),
                              np.array(list(rays.M)),
                              np.array(list(rays.N))])

            normal = np.array([np.array(list(rays.l2)),
                               np.array(list(rays.m2)),
                               np.array(list(rays.n2))])

            # rotate into global coordinates
            if global_coords == True:
                position = offset + Rmat @ position
                angle = Rmat @ angle
                normal = Rmat @ normal

            # Filter the values at the end because ZOS allocates extra space
            position = position[:,:self.Px.shape[-1]]
            angle = angle[:,:self.Px.shape[-1]]
            normal = normal[:,:self.Px.shape[-1]]
            OPD = np.array(list(rays.opd))
            print('opd = ',OPD.shape)
            OPD = OPD[:self.Px.shape[-1]]
            print('opd = ',OPD.shape)

            # convert to numpy arrays
            self.xData.append(position[0,:])
            self.yData.append(position[1,:])
            self.zData.append(position[2,:])

            self.lData.append(angle[0,:])
            self.mData.append(angle[1,:])
            self.nData.append(angle[2,:])

            print(angle[2,:].shape)

            self.l2Data.append(normal[0,:])
            self.m2Data.append(normal[1,:])
            self.n2Data.append(normal[2,:])

            self.R.append(Rmat)
            self.O.append(offset)
            self.opd.append(OPD)

            # always close your tools
            tool.Close()
        print('Raytrace Completed!')


    def ConvertRayDataToPRTData(self):
        """Function that computes the PRT-relevant data from ray and material data
        Mathematics principally from Polarized Light and Optical Systems by Chipman, Lam, Young 2018
        """

        # Compute AOI
        self.aoi = []
        self.kout = []
        self.kin = []
        self.norm = []
        # normal vector
        for i in range(len(self.surflist)):

            lData = self.lData[i]
            mData = self.mData[i]
            nData = self.nData[i]

            l2Data = self.l2Data[i]
            m2Data = self.m2Data[i]
            n2Data = self.n2Data[i]

            # Maintain right handed coords to stay with Chipman sign convention
            norm = -np.array([l2Data,m2Data,n2Data])
            total_rays_in_both_axes = self.xData[i].shape[0]

            # convert to angles of incidence
            # calculates angle of exitance from direction cosine
            # the LMN direction cosines are for AFTER refraction
            # need to calculate via Snell's Law the angle of incidence
            numerator = (lData*l2Data + mData*m2Data + nData*n2Data)
            denominator = ((lData**2 + mData**2 + nData**2)**0.5)*(l2Data**2 + m2Data**2 + n2Data**2)**0.5
            aoe_data = np.arccos(-numerator/denominator)
            # aoe = aoe_data - (aoe_data[0:total_rays_in_both_axes] > np.pi/2) * np.pi # don't really know what this is doing
            aoe = aoe_data

            # Compute kin with Snell's Law: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
            self.kout.append(np.array([lData,mData,nData])/np.sqrt(lData**2 + mData**2 + nData**2))

            if self.mode == 'transmission':
                # Snell's Law
                self.aoi.append((np.arcsin(self.n2/self.n1 * np.sin(aoe))))
                self.kin.append(np.cos(np.arcsin(self.n2*np.sin(np.arccos(self.kout[i]))/self.n1)))

            elif self.mode == 'reflection':
                # Snell's Law
                self.aoi.append(aoe)
                self.kin.append(self.kout[i] - 2*np.cos(self.aoi[i])*norm)
                # print('max angle = ',max(-aoe).all()*180/np.pi)

            self.norm.append(-np.array([l2Data,m2Data,n2Data])/np.sqrt(l2Data**2 + m2Data**2 + n2Data**2))

    def ComputePRTMatrix(self):

        """Computes the PRT matrix per ray given the PRT data
        Requires that ConvertRayDataToPRTData() is executed
        FUTURE: Make parent function that calls both so user doesn't have to
        """

        # print(self.kin[0].shape)
        self.P = []
        self.J = []
        print(self.stack)
        for j in range(len(self.surflist)):
            Pmat = np.empty([3,3,self.kin[0].shape[1]],dtype='complex128')
            Jmat = np.empty([3,3,self.kin[0].shape[1]],dtype='complex128')

            # negate the surface normal to maintain handedness of coordinate system
            for i in range(self.kin[j].shape[1]):
                Pmat[:,:,i],Jmat[:,:,i] = pol.ConstructPRTMatrix(self.kin[j][:,i],
                                                        self.kout[j][:,i],
                                                        self.norm[j][:,i],
                                                        self.aoi[j][i],
                                                        self.n1,self.n2,
                                                        self.wavelength,
                                                        recipe=self.stack)
            self.P.append(Pmat)
            self.J.append(Jmat)

    def ComputeTotalPRTMatrix(self):

        """Computes effective PRT Matrix for the entiresystem
        """

        for j in range(len(self.surflist)):

            if j == 0:

                self.Ptot = self.P[j]

            else:

                self.Ptot = mat.MatmulList(self.P[j],self.Ptot)

    def PRTtoJonesMatrix(self,aloc,exit_x):

        """_summary_
        """

        # initialize Jtot
        self.Jtot = np.empty(self.Ptot.shape,dtype='complex128')

        for i in range(self.Ptot.shape[-1]):
            # 
            self.Jtot[:,:,i] = pol.GlobalToLocalCoordinates(self.Ptot[:,:,i],self.kin[0][:,i],self.kout[-1][:,i],a=aloc,exit_x=exit_x)

    def WriteTotalPRTMatrix(self,filename):
        """Writes the PRT matrix to a FITS file
        TODO: add wavelength, system file path to header
        WARNING: defaults to overwrite=True

        Parameters
        ----------
        filename : str
            filename for the FITS file to save
        """
        
        write.WriteMatrixToFITS(self.Ptot,filename)
    
    def WriteTotalJonesMatrix(self,filename):
        """Writes the Jones matrix to a FITS file
        TODO: add wavelength, system file path to header
        WARNING: defaults to overwrite=True

        Parameters
        ----------
        filename : str
            filename for the FITS file to save
        """
        
        write.WriteMatrixToFITS(self.Jtot,filename)
        
    def WriteDiaRetAoi(self,filename):
        from astropy.io import fits
        """
        Writes to a fits file of diattenuation and retardance computed per surface.
        The output format is an npix x npix x Nsur x 3 array where the last axis is
        0: AOI
        1: Diattenuation
        2: Retardance
        TODO:
        3: x of 1st eigenpolarization
        4: y of 1st eigenpolarization
        5: x of 2nd eigenpolarization
        6: y of 2nd eigenpolarization
        """
        
        npix = int(np.sqrt(self.Ptot.shape[-1]))
        print('npix = ',npix)
        
        poldata = np.empty([npix,npix,len(self.P),3])
        
        # Loop over surface index 
        for i in range(poldata.shape[-2]):
            
            # Clear the diat and ret boxes
            D = np.empty(self.Ptot.shape[-1])
            R = np.empty(self.Ptot.shape[-1])
            
            # Loop over pixels
            for j in range(self.P[0].shape[-1]):
                
                # Compute Diattenuation, Retardance, Eigenpolarizations from SVD
                # D[j],R[j] = pol.ComputeDRFromPRT(self.P[i][:,:,j])
                D[j],R[j] = pol.ComputeDRFromAOI(self.aoi[i][j],self.n1,self.n2)
                
                
                
            # Load Angle of Incidence
            poldata[:,:,i,0] = np.reshape(self.aoi[i],[npix,npix])
            
            # Load the Diattenuation, retardance
            poldata[:,:,i,1] = np.reshape(D,[npix,npix])
            poldata[:,:,i,2] = np.reshape(R,[npix,npix])
        
        # Now write to fits file
        hdu = fits.PrimaryHDU(poldata)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename,overwrite=True)
        
    def MarchRayfront(self,dis,surf=-1):
        
        # Positions
        x = self.xData[surf]
        y = self.yData[surf]
        z = self.zData[surf]
        
        # Angles
        l = self.lData[surf]
        m = self.mData[surf]
        n = self.nData[surf]
        
        # arrange into vectors
        r = np.array([x,y,z])
        k = np.array([l,m,n])
        
        # propagate
        r_prime = r + k*dis
        
        # change the positions
        self.xData[surf] = r_prime[0,:]
        self.yData[surf] = r_prime[1,:]
        self.zData[surf] = r_prime[2,:]

    # def ComputeOPD(self):

    #     # Iterate through ray coordinates and use distance formula to compute OPD
    #     self.opd = np.empty(self.xData[0].shape)

    #     for i in range(len(self.xData)-1):

    #         xo = self.xData[i]
    #         yo = self.yData[i]
    #         zo = self.zData[i]

    #         xp = self.xData[i+1]
    #         yp = self.yData[i+1]
    #         zp = self.zData[i+1]

    #         self.opd += np.sqrt((xp-xo)**2 + (yp-yo)**2 + (zp-zo)**2)




                


            

            


            


