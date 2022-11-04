import numpy as np
import zosapi
import poke.polarization as pol
import poke.poke_math as mat
import poke.writing as write
from poke.gbd import * 
import poke.thinfilms_prysm as tf

def TraceThroughZOS(raysets,pth,surflist,nrays,wave,global_coords):

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
    xData = np.empty([len(raysets),len(surflist),maxrays])
    yData = np.empty([len(raysets),len(surflist),maxrays])
    zData = np.empty([len(raysets),len(surflist),maxrays])

    lData = np.empty([len(raysets),len(surflist),maxrays])
    mData = np.empty([len(raysets),len(surflist),maxrays])
    nData = np.empty([len(raysets),len(surflist),maxrays])

    l2Data = np.empty([len(raysets),len(surflist),maxrays])
    m2Data = np.empty([len(raysets),len(surflist),maxrays])
    n2Data = np.empty([len(raysets),len(surflist),maxrays])

    opd = np.empty([len(raysets),len(surflist),maxrays])

    # Don't need these yet
    # # The global rotation matrix
    # R = []

    # # The global offset vector
    # O = []

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

            OPD = np.array(list(rays.opd))

            # rotate into global coordinates
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
        # total_rays_in_both_axes = int(LData[0].shape

        # convert to angles of incidence
        # calculates angle of exitance from direction cosine
        # the LMN direction cosines are for AFTER refraction
        # need to calculate via Snell's Law the angle of incidence
        numerator = (lData*l2Data + mData*m2Data + nData*n2Data)
        denominator = ((lData**2 + mData**2 + nData**2)**0.5)*(l2Data**2 + m2Data**2 + n2Data**2)**0.5
        aoe_data = np.arccos(-numerator/denominator) # now in radians
        # aoe = aoe_data - (aoe_data[0:total_rays_in_both_axes] > np.pi/2) * np.pi # don't really know what this is doing
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

        # negate the surface normal to maintain handedness of coordinate system
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


        


        


