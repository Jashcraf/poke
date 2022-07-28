# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:08:11 2022

@author: qjare
"""

import numpy as np
import cmath as cm

def rotation3D(angle,axis):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[(1-c)*axis[0]**2 + c, (1-c)*axis[0]*axis[1] - s*axis[2], (1-c)*axis[0]*axis[2] + s*axis[1]],
                    [(1-c)*axis[1]*axis[0] + s*axis[2], (1-c)*axis[1]**2 + c, (1-c)*axis[1]*axis[2] - s*axis[0]],
                    [(1-c)*axis[2]*axis[0] - s*axis[1], (1-c)*axis[1]*axis[2] + s*axis[0], (1-c)*axis[2]**2 + c]])
    return mat

def vectorAngle(u,v):
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    if u@v<0:
        return np.pi - 2*np.arcsin(np.linalg.norm(-v-u)/2)
    else:
        return 2*np.arcsin(np.linalg.norm(v-u)/2)

a = np.array([0,1,0])
x0=np.array([1,0,0])

a=a/np.linalg.norm(a)
x0=x0/np.linalg.norm(x0)

kx=1
ky=2
kz=10

k=np.array([kx,ky,kz])
r=np.cross(k,a)

k=k/np.linalg.norm(k)
r=r/np.linalg.norm(r)

theta = -vectorAngle(k,a)

rot=rotation3D(theta,r)
det=np.linalg.det(rot)
print('Good rotation matrix:')
print(rot)
print('Determinant = '+str(det))

x = rot @ x0
y = rot @ np.cross(a,x0)

O_x = np.array([[x[0],y[0],k[0]],
                [x[1],y[1],k[1]],
                [x[2],y[2],k[2]]])

print('Ox from Quinns Code = ')
print(np.linalg.inv(O_x))
print('Determinant = '+str(np.linalg.det(np.linalg.inv(O_x))))

x = np.array([1-k[0]**2/(1+k[1]),
                -k[0],
                -k[0]*k[2]/(1+k[1])])
y = np.array([k[0]*k[2]/(1+k[1]),
                k[2],
                k[2]**2/(1+k[1]) -1])

O_x = np.array([[x[0],y[0],k[0]],
                [x[1],y[1],k[1]],
                [x[2],y[2],k[2]]])

print('Ox from Textbook = ')
print(np.linalg.inv(O_x))
print('Determinant = '+str(np.linalg.det(np.linalg.inv(O_x))))

# the first ky in the [0,0] element of this matrix may or may not be a 1. There is a discrepancy between eq. 11.13 and 11.14 in the textbook, but neither way works
# rotTextbook = np.array([[ky-(kx**2)/(1+ky), kx, -(kx*ky)/(1+ky)],
#                         [-kx,ky,-kz],
#                         [-(kx*kz)/(1+ky),kz,1-(kz**2)/(1+ky)]])
# print('Bad rotation matrix:')
# print(rotTextbook)
