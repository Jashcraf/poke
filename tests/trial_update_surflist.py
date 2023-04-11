pth = 'C:/Users/UASAL-OPTICS/Desktop/poke/tests/Hubble_Test.zmx'
nrays = 32
wave = 1
global_coords = True

n_Al = 1.2 + 1j*7.115 # 600nm from CV Al coating MUL
n_Al = n_Al #np.complex64(n_Al)

s1 = {
    'surf':2,
    'coating':n_Al,
    'mode':'reflect'
}

s2 = {
    'surf':4,
    'coating':n_Al,
    'mode':'reflect'
}

s3 = {
    'surf':8,
    'coating':n_Al,
    'mode':'reflect'
}

si = {
    'surf':11,
    'coating':1,
    'mode':'reflect'
}