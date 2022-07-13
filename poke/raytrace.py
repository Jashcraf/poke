import numpy as np
import zosapi
import poke.poke_core as pol

class RayBundle:

    def __init__(self,nrays,n1,n2,mode='reflection'):

        # number of rays across a grid
        self.nrays = nrays

        # Want to add support for accepting lists of these items, for now they are singular
        self.n1 = n1
        self.n2 = n2
        self.mode = mode

        # NormUnPol ray coordinates
        x = np.linspace(-1,1,nrays)
        x,y = np.meshgrid(x,x)
        X = np.ravel(x)
        Y = np.ravel(y)
        self.Px = np.ravel(x)[X**2 + Y**2 <= 1]
        self.Py = np.ravel(y)[X**2 + Y**2 <= 1]

        self.Hx = np.zeros(self.Px.shape)
        self.Hy = np.zeros(self.Py.shape)

    def TraceThroughZOS(self,pth,surflist,wave=1):

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
            position = offset + Rmat @ position
            angle = Rmat @ angle
            normal = Rmat @ normal
            

            # convert to numpy arrays
            self.xData.append(position[0,:])
            self.yData.append(position[1,:])
            self.zData.append(position[2,:])

            self.lData.append(angle[0,:])
            self.mData.append(angle[1,:])
            self.nData.append(angle[2,:])

            self.l2Data.append(normal[0,:])
            self.m2Data.append(normal[1,:])
            self.n2Data.append(normal[2,:])

        # always close your tools
        tool.Close()
        print('Raytrace Completed!')

    def ConvertRayDataToPRTData(self):

        # Compute AOI
        self.aoi = []
        self.kout = []
        self.kin = []
        # normal vector
        for i in range(len(self.surflist)):

            lData = self.lData[i]
            mData = self.mData[i]
            nData = self.nData[i]

            l2Data = self.l2Data[i]
            m2Data = self.m2Data[i]
            n2Data = self.n2Data[i]


            norm = np.array([lData,mData,nData])
            norm /= np.abs(np.linalg.norm(norm))
            total_rays_in_both_axes = self.xData[i].shape[0]

            # convert to angles of incidence
            # calculates angle of exitance from direction cosine
            # the LMN direction cosines are for AFTER refraction
            # need to calculate via Snell's Law the angle of incidence
            numerator = (lData*l2Data + mData*m2Data + nData*n2Data)
            denominator = ((lData**2 + mData**2 + nData**2)**0.5)*(l2Data**2 + m2Data**2 + n2Data**2)**0.5
            aoe_data = np.arccos(numerator/denominator)
            aoe = aoe_data - (aoe_data[0:total_rays_in_both_axes] > np.pi/2) * np.pi # don't really know what this is doing
            # aoe = np.abs(aoe)

            # Compute kin with Snell's Law: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
            self.kout.append(np.array([lData,mData,nData])/np.sqrt(lData**2 + mData**2 + nData**2))

            if self.mode == 'transmission':
                # Snell's Law
                self.aoi.append(np.abs(np.arcsin(self.n2/self.n1 * np.sin(aoe))))
                self.kin.append(np.cos(np.arcsin(self.n2*np.sin(np.arccos(self.kout[i]))/self.n1)))

            elif self.mode == 'reflection':
                # Snell's Law
                self.aoi.append(-aoe)
                self.kin.append(self.kout[i] - 2*np.cos(self.aoi[i])*norm)
                # print('max angle = ',max(-aoe).all()*180/np.pi)


            

            


            


