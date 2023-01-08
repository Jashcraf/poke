from poke.thinfilms_prysm import characteristic_matrix_p
from poke.thinfilms_prysm import *
import numpy as np
import matplotlib.pyplot as plt

# Test for p-polarized light

wlen = 0.75e-6
d = 20e-9*(1+0.01*np.random.randn(32,32))
d = np.ravel(d)
n1 = 1.4542*(1+0.01*np.random.randn(32,32)) # SiO2 at 750nm
n1 = np.ravel(n1)
th = 60 # deg

to_rad = np.pi/180

n0 = 1
n2 = 0.031165 + 1j*5.1949 # Ag at 750nm

# first two indices are x,y position
stack = np.ones([2,d.shape[-1],d.shape[-1]])
stack[0,:] = n1
stack[1,:] = d

r,t = multilayer_stack_rt(stack,wlen,'p',aoi=th)
rs = np.reshape(r,[32,32])
ts = np.reshape(t,[32,32])

plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(np.abs(rs))
plt.title('Magnitude')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.angle(rs))
plt.title('Phase')
plt.colorbar()
plt.show()

