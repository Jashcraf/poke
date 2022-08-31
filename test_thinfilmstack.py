import numpy as np
import poke.thinfilms as tf

FREESPACE_IMPEDANCE = 376.730313668 # ohms
FREESPACE_IMPEDANCE_INV = 1/FREESPACE_IMPEDANCE

aoi = 0 # normal incidence!
n1 = 1

# Thin Film Stuff
n_Ag = 0.12525718 + 1j*3.7249341450547577 
n_SiN = 2.00577335
t_Ag = 110e-9 #* 1e6
n_ZD = 1.5418
t_ZD = 50e-3 #* 1e6
t_SiN = 0.0085e-6 #* 1e6
wl = 600e-9

# Angle in zerodur layer
aor = np.arcsin(n1*np.sin(aoi)/n_ZD)

# Characteristic admittance of the substrate
eta_medium_s =  FREESPACE_IMPEDANCE_INV * n_ZD * np.cos(aor)
eta_medium_p =  FREESPACE_IMPEDANCE_INV * n_ZD / np.cos(aor)

# Characteristic admittance of vacuum
eta0_s = FREESPACE_IMPEDANCE * np.cos(aoi)
eta0_p = FREESPACE_IMPEDANCE / np.cos(aoi)

# set up the vectors
eta_vec_s = np.array([1,eta_medium_s])
eta_vec_p = np.array([1,eta_medium_p])


stack = [(n_SiN,t_SiN),(n_Ag,t_Ag)]

## Matrix for SiN layer
# Snell's Law to aoi in film
aoi = np.arcsin(n1*np.arcsin(aoi)/n_SiN)

# Phase thickness of film
B = 2*np.pi * n_SiN * t_SiN * np.cos(aoi) / wl

# Characteristic admittance of film, s
es = FREESPACE_IMPEDANCE_INV * n_SiN * np.cos(aoi)

# Characteristic admittance of film, p
ep = FREESPACE_IMPEDANCE_INV * n_SiN / np.cos(aoi)

# Characteristic matrix of film, s
CS_SiN = np.array([[np.cos(B),1j*np.sin(B)/es],[1j*es*np.sin(B),np.cos(B)]])
CP_SiN = np.array([[np.cos(B),1j*np.sin(B)/ep],[1j*ep*np.sin(B),np.cos(B)]])

## Matrix for Ag layer
# Snell's Law to aoi in film
aoi = np.arcsin(n_SiN*np.arcsin(aoi)/n_Ag)

# Phase thickness of film
B = 2*np.pi * n_Ag * t_Ag * np.cos(aoi) / wl

# Characteristic admittance of film, s
es = FREESPACE_IMPEDANCE_INV * n_Ag * np.cos(aoi)

# Characteristic admittance of film, p
ep = FREESPACE_IMPEDANCE_INV * n_Ag / np.cos(aoi)

# Characteristic matrix of film, s
CS_Ag = np.array([[np.cos(B),1j*np.sin(B)/es],[1j*es*np.sin(B),np.cos(B)]])
CP_Ag = np.array([[np.cos(B),1j*np.sin(B)/ep],[1j*ep*np.sin(B),np.cos(B)]])

## Compute Characteristic Matrix
CS_mat = CS_Ag @ CS_SiN 
CP_mat = CP_Ag @ CP_SiN

## Compute B and C coefficients
BC_s = CS_mat @ np.array([1,eta_medium_s])
BC_p = CP_mat @ np.array([1,eta_medium_p])

## Compute Reflection
r_s = (eta0_s*BC_s[0] - BC_s[1])/(eta0_s*BC_s[0] + BC_s[1])
r_p = (eta0_p*BC_p[0] - BC_p[1])/(eta0_p*BC_p[0] + BC_p[1])

# Now test the function
rs,ts,rp,tp = tf.ComputeThinFilmCoeffsCLY(stack,aoi,wl)

print('Test Values')
print(rs)
print(rp)

print('"Truth" Values')
print(r_s)
print(r_p)
