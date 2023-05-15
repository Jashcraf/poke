from poke.poke_math import np
import poke.polarization as pol
import poke.poke_math as mat
import poke.thinfilms as tf

def TraceThroughZOS(raysets,pth,surflist,nrays,wave,global_coords):

    import zosapi

    """Traces initialized rays through a zemax opticstudio file

    Parameters
    ----------
    raysets : np.ndarray
        4 x Nrays array containing normalized pupil coordinates and field coordinates. Structure is like
        [x1,x2,...,xN]
        [y1,y2,...,yN]
        [l1,l2,...,lN]
        [m1,m2,...,mN]

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

    Returns
    -------
    positions : list of ndarrays
        list containing [xData,yData,zData]. Each array contains positions indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    
    directions : list
        list containing [lData,mData,nData]. Each array contains direction cosines indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    normals: list
        list containing [l2Data,m2Data,n2Data]. Each array contains surface normals indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    opd : ndarray
        Array containing the total optical path of a ray indexed by
        0 = rayset
        1 = surface
        2 = coordinate

    """

    import zosapi
    from System import Enum,Int32,Double,Array
    import clr,os
    dll = os.path.join(os.path.dirname(os.path.realpath(__file__)),r'Raytrace.dll')
    clr.AddReference(dll)

    import BatchRayTrace

    zos = zosapi.App()
    TheSystem = zos.TheSystem
    ZOSAPI = zos.ZOSAPI
    TheSystem.LoadFile(pth,False)

    # Check to make sure the ZOSAPI is working
    if TheSystem.LDE.NumberOfSurfaces < 4:
        print('File was not loaded correctly')
        exit()
    
    if surflist[-1]['surf'] > TheSystem.LDE.NumberOfSurfaces:
        print('last surface > num surfaces, setting last surface to num surfaces')
        surflist[-1]['surf'] = TheSystem.LDE.NumberOfSurfaces

    maxrays = raysets[0].shape[-1]

    # Dimension 0 is ray set, Dimension 1 is surface, dimension 2 is coordinate
    # Satisfies broadcasting rules!
    xData = np.empty([len(raysets),len(surflist),maxrays])
    yData = np.empty([len(raysets),len(surflist),maxrays])
    zData = np.empty([len(raysets),len(surflist),maxrays])

    lData = np.empty([len(raysets),len(surflist),maxrays])
    mData = np.empty([len(raysets),len(surflist),maxrays])
    nData = np.empty([len(raysets),len(surflist),maxrays])

    l2Data = np.empty([len(raysets),len(surflist),maxrays])
    m2Data = np.empty([len(raysets),len(surflist),maxrays])
    n2Data = np.empty([len(raysets),len(surflist),maxrays])

    # Necessary for GBD calculations, might help PRT calculations
    opd = np.empty([len(raysets),len(surflist),maxrays])

    for rayset_ind,rayset in enumerate(raysets):

        # Get the normalized coordinates
        Px = rayset[0]
        Py = rayset[1]
        Hx = rayset[2]
        Hy = rayset[3]

        for surf_ind,surfdict in enumerate(surflist):

            surf = surfdict['surf']
            
            # Some ZOS-API setup
            tool = TheSystem.Tools.OpenBatchRayTrace()
            normUnpol = tool.CreateNormUnpol(maxrays, ZOSAPI.Tools.RayTrace.RaysType.Real, surf)
            reader = BatchRayTrace.ReadNormUnpolData(tool, normUnpol)
            reader.ClearData()

            # THIS OVER-INITIALIZES THE NUMBER OF RAYS. 
            # This initialization is weird because it requires allocating space for a square of rays
            # so there will be extra which we remove later. 
            rays = reader.InitializeOutput(nrays)

            # Add rays to reader
            reader.AddRay(wave,Hx,Hy,Px,Py,
                        Enum.Parse(ZOSAPI.Tools.RayTrace.OPDMode,'None'))

            isfinished = False

            # Read rays
            while not isfinished:
                segments = reader.ReadNextBlock(rays)
                if segments == 0:
                    isfinished = True

            # Global Coordinate Conversion
            # Have to pre-allocate a sysDbl for this method to execute
            sysDbl = Double(1.0)
            success,R11,R12,R13,R21,R22,R23,R31,R32,R33,XO,YO,ZO = TheSystem.LDE.GetGlobalMatrix(int(surf),
                                                                                                sysDbl,sysDbl,sysDbl,
                                                                                                sysDbl,sysDbl,sysDbl,
                                                                                                sysDbl,sysDbl,sysDbl,
                                                                                                sysDbl,sysDbl,sysDbl)
            # Did the raytrace succeed?
            if success != 1:
                print('Ray Failure at surface {}'.format(surf))

            # Global Rotation Matrix
            Rmat = np.array([[R11,R12,R13],
                            [R21,R22,R23],
                            [R31,R32,R33]])

            position = np.array([np.array(list(rays.X)),
                                np.array(list(rays.Y)),
                                np.array(list(rays.Z))])

            # I think this is just per-surface so it doesn't really need to be a big list, just a single surface.
            # TODO: Change later when cleaning up the code
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

            OPD = np.array(list(rays.opd))

            # rotate into global coordinates - necessary for PRT
            if global_coords == True:
                print('tracing with global coordinates')
                position = offset + Rmat @ position
                angle = Rmat @ angle
                normal = Rmat @ normal

            # Filter the values at the end because ZOS allocates extra space
            position = position[:,:maxrays]
            angle = angle[:,:maxrays]
            normal = normal[:,:maxrays]
            OPD = OPD[:maxrays]

            # Append data to lists along the surface dimension
            xData[rayset_ind,surf_ind] = position[0]
            yData[rayset_ind,surf_ind] = position[1]
            zData[rayset_ind,surf_ind] = position[2]

            lData[rayset_ind,surf_ind] = angle[0]
            mData[rayset_ind,surf_ind] = angle[1]
            nData[rayset_ind,surf_ind] = angle[2]

            l2Data[rayset_ind,surf_ind] = normal[0]
            m2Data[rayset_ind,surf_ind] = normal[1]
            n2Data[rayset_ind,surf_ind] = normal[2]

            # I don't think we need R and O, but might be useful to store just in case. Commenting out for now
            # R.append(Rmat)
            # O.append(offset)
            opd[rayset_ind,surf_ind] = OPD

            # always close your tools
            tool.Close()

    # This isn't necessary but makes the code more readable
    positions = [xData,yData,zData]
    directions = [lData,mData,nData]
    normals = [l2Data,m2Data,n2Data]

    # Just a bit of celebration
    print('{nrays} Raysets traced through {nsurf} surfaces'.format(nrays=rayset_ind+1,nsurf=surf_ind+1))
    
    # And finally return everything
    return positions,directions,normals,opd

def TraceThroughCV(raysets,pth,surflist,nrays,wave,global_coords,global_coord_reference='1'):

    # Code V Imports for com interface
    import sys
    from pythoncom import (CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED, com_error )
    from pythoncom import VT_VARIANT, VT_BYREF, VT_ARRAY, VT_R8
    from win32com.client import DispatchWithEvents, Dispatch, gencache, VARIANT
    from win32com.client import constants as c  # To use enumerated constants from the COM object typelib
    from win32api import FormatMessage

    sys.coinit_flags = COINIT_MULTITHREADED
    dir = "c:\cvuser"

    # Class to instantiate CV interface
    class ICVApplicationEvents:
        def OnLicenseError(self, error):
            # This event handler is called when a licensing error is 
            # detected in the CODE V application.
            print ("License error: %s " % error)

        def OnCodeVError(self, error):
            # This event handler is called when a CODE V error message is issued
            print ("CODE V error: %s " % error)

        def OnCodeVWarning(self, warning):
            # This event handler is called when a CODE V warning message is issued
            print ("CODE V warning: %s " % warning)

        def OnPlotReady(self, filename, plotwindow):
            # This event handler is called when a plot file, refered to by filename,
            # is ready to be displayed.
            # The event handler is responsible for saving/copying the
            # plot data out of the file specified by filename
            print ("CODE V Plot: %s in plot window %d" % (filename ,plotwindow) )

    zoompos = 1
    wavelen = 1
    fieldno = 0
    refsurf = 0
    ray_ind = 0
        
    cv = DispatchWithEvents("CodeV.Application", ICVApplicationEvents)
    cv.StartCodeV()

    # Load the file
    print(f'res {pth}')
    cv.Command(f'res {pth}')

    # clear any existing buffer data
    cv.Command('buf n')      # turn off buffer saving if it exists
    cv.Command('buf del b0') # clear the buffer

    # Set wavelength to 1um so OPD are in units of um
    cv.Command('wl w1 1000')
    
    # Set up global coordinate reference
    if global_coords:
        cv.Command(f'glo s{global_coord_reference} 0 0 0')
        # cv.Command(f'pol y')
        print(f'global coordinate reference set to surface {global_coord_reference}')
        offset_rows = 7
        offset_columns = 0 # for apertured systems
    else:
        cv.Command('glo n')
        offset_rows = -1+7

    # Configure ray output format to get everything we need for PRT/GBD
    cv.Command('rof x y z l m n srl srm srn aoi aor')

    # How many surfaces do we have?
    numsurf = int(cv.EvaluateExpression('(NUM S)'))
    assert numsurf >= 3
    print('number of surfaces = ',numsurf)

    maxrays = raysets[0].shape[-1]

    # Dimension 0 is ray set, Dimension 1 is surface, dimension 2 is coordinate
    # Satisfies broadcasting rules!
    xData = np.empty([len(raysets),len(surflist),maxrays])
    yData = np.empty([len(raysets),len(surflist),maxrays])
    zData = np.empty([len(raysets),len(surflist),maxrays])

    lData = np.empty([len(raysets),len(surflist),maxrays])
    mData = np.empty([len(raysets),len(surflist),maxrays])
    nData = np.empty([len(raysets),len(surflist),maxrays])

    l2Data = np.empty([len(raysets),len(surflist),maxrays])
    m2Data = np.empty([len(raysets),len(surflist),maxrays])
    n2Data = np.empty([len(raysets),len(surflist),maxrays])

    # Necessary for GBD calculations, might help PRT calculations
    opd = np.empty([len(raysets),len(surflist),maxrays])
    print(surflist)


    for rayset_ind,rayset in enumerate(raysets):

        # Get the normalized coordinates
        Px = rayset[0]
        Py = rayset[1]
        Hx = rayset[2]
        Hy = rayset[3]

        for ray_ind,(px,py,hx,hy) in enumerate(zip(Px,Py,Hx,Hy)):

            # TODO : Make compatible with different wavelengths and zooms
            # out = cv.RAYRSI(zoompos,wavelen,fieldno,refsurf,[px,py,hx,hy])
            
            cv.Command('buf y')
            cv.Command(f'RSI {px} {py} {hx} {hy}')
            cv.Command('buf n')
            # if out != 0:
            #     print('raytrace failure')
            
            # TODO figure out why negated 
            fac = -1

            for surf_ind,surfdict in enumerate(surflist):

                # print(rayset_ind)
                # print('surf ind = ',surf_ind)
                # print(ray_ind)
                # print('-------')

                surf = surfdict['surf'] # surface in LDE
                # print(surf)
                # Do this the buffer scraping way
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{2+offset_columns}')
                xData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression("(BUF.NUM)")
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{3+offset_columns}')
                yData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{4+offset_columns}')
                zData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')


                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{5+offset_columns}')
                lData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{6+offset_columns}')
                mData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{7+offset_columns}')
                nData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                
                # apply the factor
                lData[rayset_ind,surf_ind,ray_ind] *= fac
                mData[rayset_ind,surf_ind,ray_ind] *= fac
                nData[rayset_ind,surf_ind,ray_ind] *= fac
                
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{8+offset_columns}')
                l2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{9+offset_columns}')
                m2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')
                cv.Command(f'BUF MOV B0 i{surf+offset_rows} j{10+offset_columns}')
                n2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')

                # xData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(x s{surf})')
                # yData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(y s{surf})')
                # zData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(z s{surf})')

                # lData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(l s{surf})')
                # mData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(m s{surf})')
                # nData[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(n s{surf})')

                # l2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(srl s{surf})')
                # m2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(srm s{surf})')
                # n2Data[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression(f'(srn s{surf})')
                # print(f'{surf}')
                # TODO: Check that this is returning the correct OPD
                cv.Command(f'BUF MOV B0 i{1+numsurf+offset_rows} j{2+offset_columns}')
                opd[rayset_ind,surf_ind,ray_ind] = cv.EvaluateExpression('(BUF.NUM)')

                fac *= -1
            cv.Command('buf del b0')

    # This isn't necessary but makes the code more readable
    # CODE V will default to mm, so we need to scale back to meters
    positions = [xData*1e-3,yData*1e-3,zData*1e-3]
    norm = np.sqrt(lData**2 + mData**2 + nData**2)
    lData /= norm
    mData /= norm
    nData /= norm
    directions = [lData,mData,nData]
    normals = [l2Data,m2Data,n2Data]

    # Just a bit of celebration
    print('{nrays} Raysets traced through {nsurf} surfaces'.format(nrays=rayset_ind+1,nsurf=surf_ind+1))

    # Close up
    cv.StopCodeV()
    del cv
    
    # And finally return everything, OPD needs to be converted to meters
    return positions,directions,normals,opd*1e-6

def ConvertRayDataToPRTData(LData,MData,NData,L2Data,M2Data,N2Data,surflist,ambient_index=1):
    """Function that computes the PRT-relevant data from ray and material data
    Mathematics principally from Polarized Light and Optical Systems by Chipman, Lam, Young 2018

    Parameters
    ----------

    LData : ndarray
        Direction cosine in the x direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    NData : ndarray
        Direction cosine in the y direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    NData : ndarray
        Direction cosine in the z direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    L2Data : ndarray
        Surface normal direction cosine in the x direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    N2Data : ndarray
        Surface normal direction cosine in the y direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate
    N2Data : ndarray
        Surface normal direction cosine in the z direction indexed by 
        0 = rayset
        1 = surface
        2 = coordinate

    surflist: list of dicts
        list of dictionaries that describe surfaces. Including surface number in raytrace,
        interaction mode, coating, etc.

    ambient_index : float, optional
        complex refractive index of the medium the optical system exists in. Defaults to 1

    Returns
    -------
    """

    # Pre-allocate lists to return
    aoi = []
    kout = []
    kin = []
    normal = []
    n1 = ambient_index

    # Loop over surfaces
    for surf_ind,surfdict in enumerate(surflist):

        # Do some digesting
        # directions
        lData = LData[surf_ind]
        mData = MData[surf_ind]
        nData = NData[surf_ind]

        # normals
        l2Data = L2Data[surf_ind]
        m2Data = M2Data[surf_ind]
        n2Data = N2Data[surf_ind]

        # grab the index
        if type(surfdict['coating']) == list:

            # compute coeff from last film, first location
            n2 = surfdict['coating'][-1]

        else: 
            # assume scalar
            n2 = surfdict['coating']



        # Maintain right handed coords to stay with Chipman sign convention
        norm = -np.array([l2Data,m2Data,n2Data])

        # # Compute number of rays
        # total_rays_in_both_axes = int(lData.shape[0])

        # convert to angles of incidence
        # calculates angle of exitance from direction cosine
        # the LMN direction cosines are for AFTER refraction
        # need to calculate via Snell's Law the angle of incidence
        numerator = (lData*l2Data + mData*m2Data + nData*n2Data)
        denominator = ((lData**2 + mData**2 + nData**2)**0.5)*(l2Data**2 + m2Data**2 + n2Data**2)**0.5
        aoe_data = np.arccos(-numerator/denominator) # now in radians
        # aoe = aoe_data - (aoe_data[0:total_rays_in_both_axes] > np.pi/2) * np.pi
        aoe = aoe_data

        # Compute kin with Snell's Law: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        kout.append(np.array([lData,mData,nData])/np.sqrt(lData**2 + mData**2 + nData**2))

        if surfdict['mode'] == 'transmit':

            # Snell's Law
            aoi.append((np.arcsin(n2/n1 * np.sin(aoe))))
            kin.append(np.cos(np.arcsin(n2*np.sin(np.arccos(kout[surf_ind]))/n1)))

        elif surfdict['mode'] == 'reflect':

            # Snell's Law
            aoi.append(-aoe)
            kin_norm = kout[surf_ind] - 2*np.cos(aoi[surf_ind])*norm
            kin_norm /= np.sqrt(kin_norm[0]**2 + kin_norm[1]**2 + kin_norm[2]**2)
            kin.append(kin_norm)

        else:
            print('Interaction mode not recognized')
        
        # saves normal in zemax sign convention
        normal.append(np.array([l2Data,m2Data,n2Data])/np.sqrt(l2Data**2 + m2Data**2 + n2Data**2))

    return aoi,kin,kout,normal

def ComputePRTMatrixFromRayData(aoi,kin,kout,norm,surflist,wavelength,ambient_index):

    """Computes the PRT matrix per ray given the PRT data
    Requires that ConvertRayDataToPRTData() is executed
    TODO: Make parent function that calls both so user doesn't have to

    Parameters
    ----------
    aoi : list of ndarrays
        contains ndarrays with angle of incidence. List is indexed by surface. 

    kin : list of ndarrays
        contains ndarrays with direction cosines incident on the surface. List is indexed by surface. 

    kout : list of ndarrays
        contains ndarrays with direction cosines exiting the surface. List is indexed by surface. 

    norm : list of ndarrays
        contains ndarrays with direction cosines of the surface normals. List is indexed by surface. 

    surflist: list of dicts
        list of dictionaries that describe surfaces. Including surface number in raytrace,
        interaction mode, coating, etc.

    wavelength : float
        wavelength to compute the PRT matrix at.

    ambient_index : float
        complex refractive index of the medium the optical system exists in.
    """

    # print(self.kin[0].shape)
    P = []
    J = []
    O = []

    for j,surfdict in enumerate(surflist):
        Pmat = np.empty([kin[0].shape[1],3,3],dtype='complex128')
        Jmat = np.empty([kin[0].shape[1],3,3],dtype='complex128')
        Omat = np.empty([kin[0].shape[1],3,3],dtype='complex128')

        # Want this to be broadcasted, let's get all of the shapes we need
        # j appears to be the surface index so let's use that
        # i appears to be the ray index which we whould prioritize broadcasting
        for i in range(kin[j].shape[1]):
            Pmat[i],Jmat[i] = pol.ConstructPRTMatrix(kin[j][:,i],
                                                    kout[j][:,i],
                                                    norm[j][:,i],
                                                    aoi[j][i],
                                                    surfdict,
                                                    wavelength,
                                                    ambient_index)
        P.append(Pmat) # PRT matrix
        J.append(Jmat) # Local Jones Coefficients
        # O.append(Omat) # Proper Retardance

    return P,J#,O

def ComputeTotalPRTMatrix(P):

    """Computes effective PRT Matrix for the entiresystem

    Parameters
    ----------

    P : list of ndarrays
        list of ndarrays containing polarization ray tracing matrices. Elements of P are of shape N x 3 x 3 so that matrix operations can be broadcast.

    Returns
    -------
    P_total : ndarray
        shape N x 3 x 3 ndarray containing the total polarization ray tracing matrix. N is the number of rays.
    """

    # P is a list, starts with first element in list bc it corresponds to first surface
    for j,P in enumerate(P):

        if j == 0:

            P_total = P

        else:

            P_total = P @ P_total #@ P

    return P_total

def PRTtoJonesMatrix(Ptot,kin,kout,aloc,exit_x):

    """Rotates the PRT matrix into the local coordinates of a Jones pupil

    TODO : swap array dimensions so that the operation can be broadcast instead of loop

    Parameters
    ----------

    Ptot : ndarray
        shape N x 3 x 3 ndarray containing the total polarization ray tracing matrix. N is the number of rays.

     kin : ndarray
        shape 3 x N ndarray with direction cosines incident on the entrance pupil of the optical system. 

    kout : list of ndarrays
        shape 3 x N ndarray with direction cosines exiting the exit pupil of the optical system. 

    Returns
    -------
    Jtot : ndarray
        shape N x 3 x 3 ndarray containing the Jones pupil of the optical system. The elements
        Jtot[:,0,2], Jtot[:,1,2], Jtot[:,2,0], Jtot[:,2,1] should be zero.
        Jtot[:,-1,-1] should be 1
    """

    # initialize Jtot
    Jtot = np.empty(Ptot.shape,dtype='complex128')

    for i in range(Ptot.shape[0]):

        Jtot[i] =  pol.GlobalToLocalCoordinates(Ptot[i],kin[:,i],kout[:,i],aloc,exit_x)
    
    return Jtot


        


        


