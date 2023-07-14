# Poke: Integrating Ray and Diffraction Models
<img width="388" alt="image" src="https://user-images.githubusercontent.com/25557892/211158902-1df4b55b-ef2a-43aa-8986-8a156441755b.png">

[![DOI](https://zenodo.org/badge/513353061.svg)](https://zenodo.org/badge/latestdoi/513353061)
[![Documentation Status](https://readthedocs.org/projects/poke/badge/?version=latest)](https://poke.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Jashcraf/poke/branch/restructure/graph/badge.svg?token=ZE0GZ9M801)](https://codecov.io/gh/Jashcraf/poke)

Poke (pronounced /poʊˈkeɪ/ or po-kay) is a ray-based physical optics module for Python. It's named this because the name encodes the k-vector (which we are raytracing!) and the E-vector (which we are propagating!). Also I came up with the name before lunchtime, when I was craving fish from [Suisan](https://www.suisan.com/our-services/fish-market/).

Poke is a package to interface with industry-standard raytracing engines to do physical optics calculations based on ray data. The goal is to open-source the ray-based propagation physics and only rely on the raytracers (Zemax OpticStudio, CODE V, etc.) to do raytracing. The development was inspired by the need to add more propagation physics modules for Coronagraphs to expand the design space, but Poke has been used to characterize existing observatories as well.

**Presently Poke supports**:
- Polarization Ray Tracing
- Gaussian Beamlet Decomposition
- Zemax OpticStudio Optical Systems
- CODE V Optical Systems
- Multilayer Thin Film Design

**Disclaimer:** Poke is currently in very early stages of development. Documentation, unit tests, and more features are being developed and added daily. If you'd like to contribute to Poke, please open an issue to start a discussion.

If you are interested in contributing / using Poke, feel free to open an issue or contact me at jashcraft@arizona.edu.

## Installation
Poke is actively developing so we reccomend installation by cloning the repository and running `setup.py`
```
git clone https://github.com/Jashcraf/poke/
cd poke
pip install .
```

Note that we currently require the `zosapi` package that is up [on PyPi by Michael Humphreys](https://github.com/x68507/zosapi)

## Papers Published using Poke
- [1] Anche, Ashcraft, and Haffert et al. "Polarization aberrations in next-generation Giant Segmented MirrorTelescopes (GSMTs) I.Effect on the coronagraphic performance," submitted to _Astronomy & Astrophysics_ (Accepted Jan 2022).
- [2] Ashcraft, Douglas, Kim, and Riggs. "Hybrid Propagation Physics for The Design and Modeling of Astronomical Observatories: a Coronagraphic Example," submitted to _Journal of Astronomical Telescopes, Instruments, and Systems_ (in review)

**Contributors**:
- Jaren Ashcraft
- Quinn Jarecki
- Trent Brendel
- Brandon Dube
- Emory Jenkins

**Acknowledgements**:

Thanks to Dr. Max Millar-Blanchaer for inspiring the Raybundle class, and Dr. Ramya Anchce for overall helpful discussions on polarization ray tracing. Thanks to Trent Brendel, Kevin Z. Derby, and Henry Quach for helpful discussions and testing during the initial development phase. Thanks to Brandon Dube for the sage Python advice. Thanks to Kian Milani for helping test poke on supercomputer GPUs. This work was supported by a NASA Space Technology Graduate Research Opporunity.

**References**
- [1] Polarized Light and Optical Systems, by Chipman, Lam, and Young (2019)
- [2] Thin-Film Optical Filters, by Macleod (1969)
- [3] Development of a new method for the wave optical propagation of ultrashort pulses through linear optical systems, by Worku (2020)

**Other Acknowledgements**
- Suisan fish market


