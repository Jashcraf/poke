

class Config:

    def __init__(self,
                 refractive_index_sign='positive',
                 intensity_cmap='inferno',
                 phase_cmap='coolwarm',
                 real_cmap='viridis',
                 imag_cmap='plasma'):

        self.refractive_index_sign = refractive_index_sign
        self.intensity_cmap = intensity_cmap
        self.phase_cmap = phase_cmap
        self.real_cmap = real_cmap
        self.imag_cmap = imag_cmap

    @property
    def refractive_index_sign(self):
        return self._refractive_index_sign
 
    @refractive_index_sign.setter
    def refractive_index_sign(self, refractive_index_sign):

        if refractive_index_sign not in ("positive", "negative"):
            raise ValueError('Invalid sign. Please use "positive" or "negative"')

        self._refractive_index_sign = refractive_index_sign

config = Config()