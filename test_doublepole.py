import numpy as np


# Make the rotation matrix
k = np.array([0,0,1])
a = np.array([0,1,0])

th = -np.arccos(np.dot(k,a))
r = np.cross(k,a)

ux = r[0]
uy = r[1]
uz = r[2]

kx = k[0]
ky = k[1]
kz = k[2]

R11 = np.cos(th) + ux**2 *(1-np.cos(th))
R12 = ux*uy*(1-np.cos(th)) - uz*np.sin(th)
R13 = ux*uz*(1-np.cos(th)) + uy*np.sin(th)

R21 = uy*ux*(1-np.cos(th)) + uz*np.sin(th)
R22 = np.cos(th) + uy**2 *(1-np.cos(th))
R23 = uy*uz*(1-np.cos(th)) - ux*np.sin(th)

R31 = uz*ux*(1-np.cos(th)) - uy*np.sin(th)
R32 = uz*uy*(1-np.cos(th)) + ux*np.sin(th)
R33 = np.cos(th) + uz**2 * (1-np.cos(th))

R = np.array([[R11,R12,R13],
              [R21,R22,R23],
              [R31,R32,R33]])

R_test = np.array([[ky - kx**2/(1+ky),kx,-kx*ky/(1+ky)],
                   [-kx,ky,-kz],
                   [-kx*kz/(1+ky),kz,1-kz**2/(1+ky)]])

print(R)
print(R_test)