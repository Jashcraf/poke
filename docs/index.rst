.. poke documentation master file, created by
   sphinx-quickstart on Thu Jan 12 09:59:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Poke's documentation!
================================
Poke is a ray-based physical optics package that aims to better integrate raytrace models with diffraction models, and add more physics to the optical modeling pipeline. Poke was developed as a part of a NASA Space Technology Graduate Research Opportunity which emphasised modeling of astronomical telescopes equipped with coronagraphs, but is applicable to a wide range of optical modeling problems. 

Summary
------------

**What this software does:**

* Allows the users to run raytraces of optical systems in Ansys Zemax OpticStudio (Zemax) and SYNOPSYS CODE V through the Rayfront class
* Access the optical path difference (OPD) of a given wavefront computed with ray data
* Compute the polarization aberrations of an optical system with *polarization ray tracing*
* Perform physical optics propagation with *Gaussian beamlet decomposition*
* Simulate and design multilayer thin film stacks
* Permits the reading and writing of Rayfront data to a binary file for broad distribution of optical system data that isn't limited to a raytrace

**Soft Requirements**
Using Poke requires that at least one person in your workflow has access to Zemax or CODE V, which require the following packages that are not installed by default with Poke
* *Zemax*: `zosapi by Michael Humphreys <https://github.com/x68507/zosapi>`
* *CODE V*: `pywin32 by Mark Hammond et al, which may be installed by default <https://pypi.org/project/pywin32/>`
Furthermore, there are some raytracer-specific quirks with using Poke and running raytraces in Jupyter/IPython notebooks, see `Using Jupyter Notebooks & Raytracer Specifics <https://poke.readthedocs.io/en/latest/notebooks/using_ipython.html>` for guidance on navigating them.

**Getting Started**
To get started with your first Rayfront model, check out `Intro to Poke: The Rayfront <https://poke.readthedocs.io/en/latest/notebooks/rayfrontattributes.html>`_
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules
   notebooks/installation.ipynb
   notebooks/rayfrontattributes.ipynb
   notebooks/using_ipython.ipynb
   notebooks/introtopolarization.ipynb
   notebooks/jonespupils.ipynb
   notebooks/material_data.ipynb
   notebooks/thinfilm_optimization.ipynb
   notebooks/aboutpoke.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
