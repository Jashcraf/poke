# Poke: Integrating Ray and Diffraction Models
<img width="188" alt="image" src="https://user-images.githubusercontent.com/25557892/178643479-ae0c5955-2bc4-4a80-ad82-79d438f4d6c9.png">

Poke (pronounced poh-keh) is a Polarization Ray Tracing module for Python. It's named this because the name encodes the k-vector (which we are raytracing!) and the E-vector (which we are propagating!). Also I came up with the name before lunchtime, when I was craving fish from Suisan.

Poke is a package to interface with industry-standard raytracing engines to do physical optics calculations based on ray data. The goal is to open-source the ray-based propagation physics and only rely on the raytracers (Zemax OpticStudio, CODE V, etc.) to do raytracing. The development was inspired by the need to add more propagation physics modules for Coronagraphs to expand the design space, but Poke has been used to characterize existing observatories as well.

**Presently Poke supports**:
- Polarization Ray Tracing
- Gaussian Beamlet Decomposition
- Zemax OpticStudio Optical Systems

If you are interested in contributing / using Poke, feel free to open an issue or contact me at jashcraft@arizona.edu.

## Installation
Poke is actively developing so we reccomend installation by cloning the repository and running `setup.py`
```
git clone https://github.com/Jashcraf/poke/
cd poke
pip install .
```

Note that we currently require the `zosapi` package that is up [on PyPi by Michael Humphreys](https://github.com/x68507/zosapi)

**Contributors**:
- Jaren Ashcraft
- Quinn Jarecki
- Trent Brendel

**Acknowledgements**:

Thanks to Dr. Max Millar-Blanchaer for inspiring the Raybundle class, and Dr. Ramya Anchce for overall helpful discussions on polarization ray tracing. Thanks to Trent Brendel, Kevin Z. Derby, and Henry Quach for helpful discussions and testing during the initial development phase. Thanks to Brandon Dube for the sage Python advice. This work was supported by a NASA Space Technology Graduate Research Opporunity.

**References**
- [1] Polarized Light and Optical Systems, by Chipman, Lam, and Young (2019)
- [2] Thin-Film Optical Filters, by Macleod (1969)
- [3] Development of a new method for the wave optical propagation of ultrashort pulses through linear optical systems, by Worku (2020)

**Other Acknowledgements**
- Suisan fish market


