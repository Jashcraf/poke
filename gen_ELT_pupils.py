import numpy as np
import poke.poke_core as pol
import poke.raytrace as ray
import poke.plotting as plot

import sys
print(sys.executable)

# Initialize a Raybundle
nrays_across_pupil = 32
n1 = 1 # medium index
aloc = np.array([0.,-1.,0.]) # antipole location
exit_x = np.array([-1.,0.,0.]) # reference x axis
pth = "C:/Users/UASAL-OPTICS/Desktop/poke/ELT_Paper_1/ELT/ELT"
surflist = [1,3,5,8,12]

wlens_vis = [445,464,551,658] # BGVR 
wlens_nir = [806,900,1020,1220,1630,2190,3450] # IZYJHKL
wlens_mir = [4750,10500] # MN

# refractive index of silver https://refractiveindex.info using Babar and Weaver 2015 
n_Ag = [0.052000 + 1j*2.6256,
        0.052271 + 1j*2.8097,
        0.050908 + 1j*3.5855,
        0.049816 + 1j*4.4764]
        
t_Ag = 110e-9
        
n_SiN = [1.9450 + 1j*0.000425,
         1.9378 + 1j*0.000077520,
         1.9168,
         1.9022]

t_SiN = 55e-10

t_ZD = 25.4e-3/4
n_ZD = [1.5533,1.5512,1.5444,1.5394]


        
for i,wl in enumerate(wlens_vis):

    # n dimensional index of refraction, 
#     stack = np.array([[n_SiN[i],n_Ag[i],n_ZD[i]],
#                       [t_SiN*1e6,t_Ag*1e6,t_ZD*1e6]])
    stack = [(n_SiN[i],t_SiN * 1e6),(n_Ag[i],t_Ag * 1e6)] 
    # stack = [(1,n_SiN[i],n_Ag[i],n_ZD[i]),(t_SiN,t_Ag,t_ZD)] 

    # Initialize a ray bundle
    # Using the thin film borks the calculation, why?
    print(wl)
    raybundle = ray.Rayfront(nrays_across_pupil,n1,n_Ag[i],wl*1e-9,stack=stack) # 

    # run a raytrace (please forgive the API I promise I'm doing my best)
    raybundle.TraceThroughZOS(pth+'_{}.zmx'.format(wl),surflist)
    raybundle.ConvertRayDataToPRTData()
    raybundle.ComputePRTMatrix()
    raybundle.ComputeTotalPRTMatrix()
    raybundle.PRTtoJonesMatrix(aloc,exit_x)
#     plot.PlotJonesPupil(raybundle)
    plot.PRTPlot(raybundle,surf=0)
    plot.PRTPlot(raybundle,surf=1)
    plot.PRTPlot(raybundle,surf=2)
    plot.PRTPlot(raybundle,surf=3)
    plot.PRTPlot(raybundle,surf=4)
    
    # raybundle.WriteTotalJonesMatrix('ELT_Paper_1/ELT/Jones_Pupils/ELT_BareAg_{}.fits'.format(wl))
    # raybundle.WriteDiaRetAoi('ELT_Paper_1/ELT/AOI_Dia_Ret/ELT_SiNAg_DRAOI_{}.fits'.format(wl))


