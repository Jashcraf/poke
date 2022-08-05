from poke.thinfilms_prysm import characteristic_matrix_p
from poke.thinfilms_prysm import *

# Test for p-polarized light

wlen = 0.75e-6
d = 20e-9
n1 = 1.4542 # SiO2 at 750nm
th = 60 # deg

to_rad = np.pi/180

n0 = 1
n2 = 0.031165 + 1j*5.1949 # Ag at 750nm

Mp = characteristic_matrix_p(wlen,d,n1,th*to_rad)
print(Mp)

aor = snell_aor(n1,n2,th)

Ap = multilayer_matrix_p(n0,th*to_rad,[Mp],n2,aor)
print(Ap)
print('rp tot = ',rtot(Ap))
print('Rp tot = ',np.abs(rtot(Ap))**2)