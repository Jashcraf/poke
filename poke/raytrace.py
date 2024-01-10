from poke.poke_math import np
import poke.polarization as pol
import poke.poke_math as mat
import poke.thinfilms as tf
import os


def trace_through_zos(raysets, pth, surflist, nrays, wave, global_coords):
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
    from System import Enum, Int32, Double, Array
    import clr, os

    # known directory
    # dll = os.path.join(os.path.dirname(os.path.realpath(__file__)),r'RayTrace.dll')
    dll = os.path.dirname(__file__) + "\RayTrace.dll"
    clr.AddReference(dll)

    import BatchRayTrace

    zos = zosapi.App()
    TheSystem = zos.TheSystem
    ZOSAPI = zos.ZOSAPI
    TheSystem.LoadFile(pth, False)

    # Check to make sure the ZOSAPI is working
    if TheSystem.LDE.NumberOfSurfaces < 4:
        print("File was not loaded correctly")
        exit()

    if surflist[-1]["surf"] > TheSystem.LDE.NumberOfSurfaces:
        print("last surface > num surfaces, setting last surface to num surfaces")
        surflist[-1]["surf"] = TheSystem.LDE.NumberOfSurfaces

    maxrays = raysets[0].shape[-1]

    # Dimension 0 is ray set, Dimension 1 is surface, dimension 2 is coordinate
    # Satisfies broadcasting rules!
    xData = np.empty([len(raysets), len(surflist), maxrays])
    yData = np.empty([len(raysets), len(surflist), maxrays])
    zData = np.empty([len(raysets), len(surflist), maxrays])

    lData = np.empty([len(raysets), len(surflist), maxrays])
    mData = np.empty([len(raysets), len(surflist), maxrays])
    nData = np.empty([len(raysets), len(surflist), maxrays])

    l2Data = np.empty([len(raysets), len(surflist), maxrays])
    m2Data = np.empty([len(raysets), len(surflist), maxrays])
    n2Data = np.empty([len(raysets), len(surflist), maxrays])

    mask = np.empty([len(raysets), len(surflist), maxrays])

    # Necessary for GBD calculations, might help PRT calculations
    opd = np.empty([len(raysets), len(surflist), maxrays])

    for rayset_ind, rayset in enumerate(raysets):

        # Get the normalized coordinates
        Px = rayset[0]
        Py = rayset[1]
        Hx = rayset[2]
        Hy = rayset[3]

        for surf_ind, surfdict in enumerate(surflist):

            surf = surfdict["surf"]

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
            reader.AddRay(wave, Hx, Hy, Px, Py, Enum.Parse(ZOSAPI.Tools.RayTrace.OPDMode, "None"))

            isfinished = False

            # Read rays
            while not isfinished:
                segments = reader.ReadNextBlock(rays)
                if segments == 0:
                    isfinished = True

            # Global Coordinate Conversion
            # Have to pre-allocate a sysDbl for this method to execute
            # TODO: This is a properly ugly line of code
            sysDbl = Double(1.0)
            # fmt: off
            (success, 
            R11, R12, R13, \
            R21, R22, R23, \
            R31, R32, R33, \
            XO, YO, ZO) = TheSystem.LDE.GetGlobalMatrix(
                int(surf), 
                sysDbl, sysDbl, sysDbl, sysDbl,
                sysDbl, sysDbl, sysDbl, sysDbl,
                sysDbl, sysDbl, sysDbl, sysDbl,
            )
            # fmt: on
            # Did the raytrace succeed?
            if success != 1:
                print("Ray Failure at surface {}".format(surf))

            # Global Rotation Matrix
            # fmt: off
            Rmat = np.array([[R11, R12, R13], 
                             [R21, R22, R23], 
                             [R31, R32, R33]])

            position = np.array(
                [np.array(list(rays.X)),
                 np.array(list(rays.Y)),
                 np.array(list(rays.Z))]
            )
            # fmt: on

            # I think this is just per-surface so it doesn't really need to be a big list, just a single surface.
            # TODO: Change later when cleaning up the code
            offset = np.zeros(position.shape)
            offset[0, :] = XO
            offset[1, :] = YO
            offset[2, :] = ZO

            # fmt: off
            angle = np.array(
                [np.array(list(rays.L)),
                 np.array(list(rays.M)),
                 np.array(list(rays.N))]
            )

            normal = np.array(
                [np.array(list(rays.l2)),
                 np.array(list(rays.m2)),
                 np.array(list(rays.n2))]
            )
            # fmt: on

            OPD = np.array(list(rays.opd))

            rays_that_passed = np.array(list(rays.vignetteCode))
            rays_that_passed = rays_that_passed[:maxrays]

            # rotate into global coordinates - necessary for PRT
            if global_coords == True:
                print("tracing with global coordinates")
                position = offset + Rmat @ position
                angle = Rmat @ angle
                normal = Rmat @ normal

            # Filter the values at the end because ZOS allocates extra space
            position = position[:, :maxrays]
            angle = angle[:, :maxrays]
            normal = normal[:, :maxrays]
            OPD = OPD[:maxrays]

            # Append data to lists along the surface dimension
            xData[rayset_ind, surf_ind] = position[0]
            yData[rayset_ind, surf_ind] = position[1]
            zData[rayset_ind, surf_ind] = position[2]

            lData[rayset_ind, surf_ind] = angle[0]
            mData[rayset_ind, surf_ind] = angle[1]
            nData[rayset_ind, surf_ind] = angle[2]

            l2Data[rayset_ind, surf_ind] = normal[0]
            m2Data[rayset_ind, surf_ind] = normal[1]
            n2Data[rayset_ind, surf_ind] = normal[2]

            # I don't think we need R and O, but might be useful to store just in case. Commenting out for now
            # R.append(Rmat)
            # O.append(offset)
            opd[rayset_ind, surf_ind] = OPD
            mask[rayset_ind, surf_ind] = rays_that_passed

            # always close your tools
            tool.Close()

    # This isn't necessary but makes the code more readable
    positions = [xData, yData, zData]
    directions = [lData, mData, nData]
    normals = [l2Data, m2Data, n2Data]

    # Just a bit of celebration
    print(
        "{nrays} Raysets traced through {nsurf} surfaces".format(
            nrays=rayset_ind + 1, nsurf=surf_ind + 1
        )
    )

    # And finally return everything
    return positions, directions, normals, opd, mask


def trace_through_cv(raysets, pth, surflist, nrays, wave, global_coords, global_coord_reference="1"):
    """trace raysets through a sequential code v optical system

    Parameters
    ----------
    raysets : numpy.ndarray
        arrays of shape Nrays x 4 that contain the normalized pupil + field data
        [Px,Py,Hx,Hy]
    pth : str
        path to .seq or .len file to trace the rays through
    surflist : list of dictionaries
        list of surfaces to trace the rays to
    nrays : int
        number of rays to trace
    wave : int
        wavelength number to trace, in order they appear in the LDE, starting from 1
    global_coords : boolean
        whether to trace rays using global or local coordinates.
    global_coord_reference : str, optional
        surface number to use as the global coordinate reference, by default '1'

    Returns
    -------
    positions,directions,normals,opd
        position, direction, surface normal, and OPD data
    """

    # Code V Imports for com interface
    import sys
    from pythoncom import CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED, com_error
    from pythoncom import VT_VARIANT, VT_BYREF, VT_ARRAY, VT_R8
    from win32com.client import DispatchWithEvents, Dispatch, gencache, VARIANT
    from win32com.client import (
        constants as c,
    )  # To use enumerated constants from the COM object typelib
    from win32api import FormatMessage

    sys.coinit_flags = COINIT_MULTITHREADED
    dir = "C:/CVUSER/"

    # Class to instantiate CV interface
    class ICVApplicationEvents:
        def OnLicenseError(self, error):
            # This event handler is called when a licensing error is
            # detected in the CODE V application.
            print("License error: %s " % error)

        def OnCodeVError(self, error):
            # This event handler is called when a CODE V error message is issued
            print("CODE V error: %s " % error)

        def OnCodeVWarning(self, warning):
            # This event handler is called when a CODE V warning message is issued
            print("CODE V warning: %s " % warning)

        def OnPlotReady(self, filename, plotwindow):
            # This event handler is called when a plot file, refered to by filename,
            # is ready to be displayed.
            # The event handler is responsible for saving/copying the
            # plot data out of the file specified by filename
            print("CODE V Plot: %s in plot window %d" % (filename, plotwindow))

    zoompos = 1
    wavelen = 1
    fieldno = 0
    refsurf = 0
    ray_ind = 0

    cv = DispatchWithEvents("CodeV.Application", ICVApplicationEvents)
    cv.StartCodeV()

    # Load the file
    if pth[-3:] == "len":
        print(f"res {pth}")
        cv.Command(f"res {pth}")
    elif pth[-3:] == "seq":
        print(f"in {pth}")
        cv.Command(f"in " + pth)

    # configure the file
    cv.Command("cd " + dir)
    cv.Command("dim m")

    # clear any existing buffer data
    cv.Command("buf n")  # turn off buffer saving if it exists
    cv.Command("buf del b0")  # clear the buffer

    # Set wavelength to 1um so OPD are in units of um TODO: This breaks refractive element tracing
    cv.Command("wl w1 1000")

    # Set up global coordinate reference
    if global_coords:
        cv.Command(f"glo s{global_coord_reference} 0 0 0")
        # cv.Command(f'pol y')
        print(f"global coordinate reference set to surface {global_coord_reference}")
    else:
        cv.Command("glo n")

    # Configure ray output format to get everything we need for PRT/GBD
    cv.Command("rof x y z l m n srl srm srn aoi aor")

    # How many surfaces do we have?
    numsurf = int(cv.EvaluateExpression("(NUM S)"))
    assert numsurf >= 3, f"number of surfaces = {numsurf}"

    maxrays = raysets[0].shape[-1]
    print("maxrays = ", maxrays)

    # Dimension 0 is ray set, Dimension 1 is surface, dimension 2 is coordinate
    # Satisfies broadcasting rules!
    xData = np.empty([len(raysets), len(surflist), maxrays])
    yData = np.empty([len(raysets), len(surflist), maxrays])
    zData = np.empty([len(raysets), len(surflist), maxrays])

    lData = np.empty([len(raysets), len(surflist), maxrays])
    mData = np.empty([len(raysets), len(surflist), maxrays])
    nData = np.empty([len(raysets), len(surflist), maxrays])

    l2Data = np.empty([len(raysets), len(surflist), maxrays])
    m2Data = np.empty([len(raysets), len(surflist), maxrays])
    n2Data = np.empty([len(raysets), len(surflist), maxrays])

    # Necessary for GBD calculations, might help PRT calculations
    opd = np.empty([len(raysets), len(surflist), maxrays])

    # Open an intermediate raytrace file

    for rayset_ind, rayset in enumerate(raysets):

        Hx, Hy = rayset[2, 0], rayset[3, 0]

        fac = -1
        for surf_ind, surfdict in enumerate(surflist):

            surf = surfdict["surf"]
            # fmt: off
            file = open(dir + "intermediate_raytrace.seq", "w")
            # Begin file construction
            # x,y,z,l,m,n,l2,m2,n2,aoi,aor
            file.write("! Define input variables\n")
            file.write(f"num ^input_array(2,{int(nrays**2)}) ^output_array(12,{int(nrays**2)}) ^input_ray(4)\n")
            file.write("num ^success \n")
            # file.write('! set up global coordinates\n')
            # file.write(f'glo s{global_coord_reference} 0 0 0\n')
            # file.write('glo y\n')
            file.write(f"^nrays == {nrays**2} \n")
            file.write(f"^nrays_across == sqrt(^nrays) \n")
            file.write("^x_start == -1\n")
            file.write("^x_end == 1\n")
            file.write("^y_start == -1\n")
            file.write("^y_end == 1\n")
            file.write("\n")
            file.write("rof x y z l m n srl srm srn aoi aor \n")

            file.write("! compute step size and construct input array\n")
            file.write("^step_size == absf(^y_start-^y_end)/(^nrays_across-1)\n")
            file.write("^x_cord == ^x_start\n")
            file.write("^y_cord == ^y_start\n")
            file.write("^next_row == 0\n")
            file.write("FOR ^iter 1 ^nrays\n")
            file.write("	^input_array(1,^iter) == ^x_cord\n")
            file.write("	^input_array(2,^iter) == ^y_cord\n")
            file.write("	! update the x_cord\n")
            file.write("	^x_cord == ^x_cord + ^step_size\n")
            file.write("	IF modf(^iter,^nrays_across) = 0\n")
            file.write("		^next_row == 1\n")
            file.write("	END IF\n")
            file.write("	IF ^next_row = 1\n")
            file.write("		! update y_cord\n")
            file.write("		^y_cord == ^y_cord + ^step_size\n")
            file.write("		! reset x_cord\n")
            file.write("		^x_cord == ^x_start\n")
            file.write("		! reset next_row\n")
            file.write("		^next_row == 0\n")
            file.write("	END IF\n")
            file.write("END FOR\n")
            file.write("\n")

            file.write("! Run RAYRSI and pass data to output_array\n")
            file.write("FOR ^iter 1 ^nrays\n")
            file.write("	^input_ray(1) == ^input_array(1,^iter)\n")
            file.write("	^input_ray(2) == ^input_array(2,^iter)\n")
            file.write(f"	^input_ray(3) == {Hx}\n")
            file.write(f"	^input_ray(4) == {Hy}\n")
            file.write("\n")
            file.write("    ! Execute RAYRSI\n")
            file.write("    ^success == RAYRSI(0,0,0,0,^input_ray)\n")
            # file.write('     rsi ^input_array(1), ^input_array(2), ^input_array(3), ^input_array(4),\n')
            file.write("\n")
            file.write("    ! Read ray data from lens database into output_array\n")
            file.write(f"    ^output_array(1,^iter) == (X S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(2,^iter) == (Y S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(3,^iter) == (Z S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(4,^iter) == (L S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(5,^iter) == (M S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(6,^iter) == (N S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(7,^iter) == (SRL S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(8,^iter) == (SRM S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(9,^iter) == (SRN S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(10,^iter) == (AOI S{surf} G{global_coord_reference})\n")
            file.write(f"    ^output_array(11,^iter) == (AOR S{surf} G{global_coord_reference})\n")

            # TODO: This defaults to the total OPD, not per-surface
            file.write(f"    ^output_array(12,^iter) == (OPD)\n")
            file.write("END FOR\n")
            file.write("\n")
            file.write("! Write output_array to text file\n")
            file.write("BUF Y\n")
            file.write("BUF DEL B1\n")
            file.write(f"BUF LEN {maxrays} \n")
            file.write("^result == ARRAY_TO_BUFFER(^output_array,1,1)\n")
            file.write('BUF EXP B1 SEP "intermediate_output.txt"\n')
            file.close()

            # fmt: on

            # Now execute the macro
            cv.Command(f"in intermediate_raytrace.seq")

            # And read the raydata
            raydata = np.genfromtxt(dir + "intermediate_output.txt", delimiter=" ")
            # raydata = np.genfromtxt(dir+"intermediate_output.txt",delimiter=' ',usecols=np.arange(0,int(nrays**2)))

            # Load into numpy arrays
            xData[rayset_ind, surf_ind] = raydata[:, 0]
            yData[rayset_ind, surf_ind] = raydata[:, 1]
            zData[rayset_ind, surf_ind] = raydata[:, 2]

            lData[rayset_ind, surf_ind] = raydata[:, 3] * fac
            mData[rayset_ind, surf_ind] = raydata[:, 4] * fac
            nData[rayset_ind, surf_ind] = raydata[:, 5] * fac

            l2Data[rayset_ind, surf_ind] = raydata[:, 6]
            m2Data[rayset_ind, surf_ind] = raydata[:, 7]
            n2Data[rayset_ind, surf_ind] = raydata[:, 8]

            opd[rayset_ind, surf_ind] = raydata[:, 11]

            fac *= -1

            # delete the files made
            os.remove(dir + "intermediate_raytrace.seq")
            os.remove(dir + "intermediate_output.txt")

    positions = [xData * 1e-3, yData * 1e-3, zData * 1e-3]  # correct for default to mm
    norm = np.sqrt(lData ** 2 + mData ** 2 + nData ** 2)
    lData /= norm
    mData /= norm
    nData /= norm

    norm = np.sqrt(l2Data ** 2 + m2Data ** 2 + n2Data ** 2)
    l2Data /= norm
    m2Data /= norm
    n2Data /= norm
    directions = [lData, mData, nData]
    normals = [l2Data, m2Data, n2Data]

    # Just a bit of celebration
    print(
        "{nrays} Raysets traced through {nsurf} surfaces".format(
            nrays=rayset_ind + 1, nsurf=surf_ind + 1
        )
    )

    # Close up
    cv.StopCodeV()
    del cv

    # Delete the intermediate files


    # And finally return everything, OPD needs to be converted to meters
    return positions, directions, normals, opd * 1e-6


def convert_ray_data_to_prt_data(LData, MData, NData, L2Data, M2Data, N2Data, surflist, ambient_index=1):
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
    for surf_ind, surfdict in enumerate(surflist):

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
        if type(surfdict["coating"]) == list:

            # compute coeff from last film, first location
            n2 = surfdict["coating"][-1]

        else:
            # assume scalar
            n2 = surfdict["coating"]

        # Maintain right handed coords to stay with Chipman sign convention
        norm = -np.array([l2Data, m2Data, n2Data])

        # # Compute number of rays
        # total_rays_in_both_axes = int(lData.shape[0])

        # convert to angles of incidence
        # calculates angle of exitance from direction cosine
        # the LMN direction cosines are for AFTER refraction
        # need to calculate via Snell's Law the angle of incidence
        numerator = lData * l2Data + mData * m2Data + nData * n2Data
        denominator = ((lData ** 2 + mData ** 2 + nData ** 2) ** 0.5) * (
            l2Data ** 2 + m2Data ** 2 + n2Data ** 2
        ) ** 0.5
        aoe_data = np.arccos(-numerator / denominator)  # now in radians
        # aoe = aoe_data - (aoe_data[0:total_rays_in_both_axes] > np.pi/2) * np.pi
        aoe = aoe_data

        # Compute kin with Snell's Law: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        kout.append(np.array([lData, mData, nData]) / np.sqrt(lData ** 2 + mData ** 2 + nData ** 2))

        if surfdict["mode"] == "transmit":

            # Snell's Law
            aoi.append((np.arcsin(n2 / n1 * np.sin(aoe))))
            kin.append(np.cos(np.arcsin(n2 * np.sin(np.arccos(kout[surf_ind])) / n1)))

        elif surfdict["mode"] == "reflect":

            # Snell's Law
            aoi.append(-aoe)
            kin_norm = kout[surf_ind] - 2 * np.cos(aoi[surf_ind]) * norm
            kin_norm /= np.sqrt(kin_norm[0] ** 2 + kin_norm[1] ** 2 + kin_norm[2] ** 2)
            kin.append(kin_norm)

        else:
            print("Interaction mode not recognized")

        # saves normal in zemax sign convention
        normal.append(
            np.array([l2Data, m2Data, n2Data]) / np.sqrt(l2Data ** 2 + m2Data ** 2 + n2Data ** 2)
        )

    return aoi, kin, kout, normal
