import numpy as np
import gbd 


def EvalGaussianField(base_rayfront,
                      Px_rayfront,Py_rayfront,Hx_rayfront,Hy_rayfront,
                      wavelength,detector_size):

    """
    Assumptions
    -----------
    - We are at the image plane in local coordinates, so image plane is z = 0
    - The base ray bundle and 4 differential ray bundles last surface is the image plane
    - Even beamlet distribution is sufficient
    - OF = 1.7 is sufficient for decomposition
    
    """
    
    # Initial Gaussian Beam Parameters
    zr = np.pi*wo**2/wavelength
    Qinv = np.array([[-1/(1j*zr),0],[0,-1/(1j*zr)]])
    Q = np.linalg.inv(Qinv)
    k = 2*np.pi/wavelength
    
    # Define the detector plane at z = 0
    # We use capital letters because this is the "global" detector
    X = np.linspace(-detsize/2,detsize/2,npix)
    X,Y = np.meshgrid(X,X)
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = X*0
    
    # And lower case letters for the "local" detector for each beamlet
    # These are the ones that are rotated
    x = X
    y = Y
    z = Z
    
    local_coordinates = np.array([x,y,z])
    
    nrays = base_rayfront.xData[0].shape[-1]
    npix  = len(X)
    
    # Empty phase array that we will populate with beamlet information
    # This has dimensions = npix**2 x nrays
    # and is recomputed at every propagation
    phase_box = np.empty([npix,nrays],dtype='complex128')
    
    # Empty amplitude array that we will populate with beamlet information
    # This has dimension = nrays 
    # and is recomputed at every propagation
    amp_box = np.empty(nrays,dtype='complex128')
    
    # First update the positions
    d_plusses,dminuses = ComputeFinitePropagation(base_rayfront,detector_size)
    
    # Set them to the far back positions
    base_rayfront.MarchRayfront(dminuses)
    Px_rayfront.MarchRayfront(dminuses)
    Py_rayfront.MarchRayfront(dminuses)
    Hx_rayfront.MarchRayfront(dminuses)
    Hy_rayfront.MarchRayfront(dminuses)
    
    
    
    for i in np.arange(0,nrays):
    
        ray = np.array([base_rayfront.xData[-1],
                        base_rayfront.yData[-1],
                        base_rayfront.zData[-1]])
        
        for j in np.arange(0,npix):
            
            # Ideally we have some form of selection over which rays we compute
            # For now just separate the differential computation from the bundles and do it per-ray
            ABCD,O = ComputeDifferentialFromRaybundles(base_rayfront,
                                                       Px_rayfront,
                                                       Py_rayfront,
                                                       Hx_rayfront,
                                                       Hy_rayfront)
                                                       
                                                       
                                                       
            # Propagate the Q parameter and guoy phase
            Qp_n = (C + D @ Qinv)
            Qp_d = np.linalg.inv(A + B @ Qinv)
            Qpinv   = Qp_n @ Qp_d
            guoy_phase = -1j*ComputeGouyPhase(Qpinv)
            
            # evaluate the field
            # This can be done outside I think?
            local_to_global = O @ local_coordinates + ray 
            
            print(local_to_global[local_to_global[-1,:] == 0])
            
            
            
                                                       
                                               
                                                       
            
            
            
        
    
    