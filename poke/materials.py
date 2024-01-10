import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

# Silica == Fused Silica
# fmt: off
avail_materials = [
    "Al", "Ag",  # metals
    "HfO2", "SiO2", "Ta2O5", "TiO2", "Nb2O5",  # Oxides
    "SiN",  # Nitrides
    "MgF2", "CaF2", "LiF",  # Fluorides
    "Silica",  # Glasses
]
# fmt: on


def get_abs_path(file):
    fullpth = Path(__file__).parent / "material_data" / file
    return fullpth


def create_index_model(material, verbose=False):
    """creates an interpolated material based on data available from refractiveindex.info

    "[refractiveindex.info] is made freely available under the CC0 1.0 Universal Public Domain Dedication. 
    This means you are free to use, modify, and distribute its content without any restrictions, 
    even for commercial purposes, no permission required."

    src: https://refractiveindex.info/about

    Parameters
    ----------
    material : str
        identifier string from avail_materials
    verbose : bool, optional
        whether to print information about the material data used, by default False

    Returns:
    --------
    index_model : callable
        1d interpolated refractive index model of the chosen material
    """

    # set lims
    n_end = None
    k_start = None

    if material not in avail_materials:
        print(f"Material {material} not recognized")
        print("Materials supported:")
        print(avail_materials)

    # Load materials - this will be a lot
    if material == "Al":
        pth = get_abs_path("Cheng_Al.csv")
        n_end = 427
        k_start = n_end + 3
    elif material == "Ag":
        pth = get_abs_path("Ciesielski_Ag.csv")
        n_end = 333
        k_start = n_end + 3
    elif material == "HfO2":
        pth = get_abs_path("Kuhaili_HfO2.csv")
    elif material == "SiO2":
        pth = get_abs_path("Lemarchand_SiO2.csv")
        n_end = 451
        k_start = n_end + 3
    elif material == "SiN":
        pth = get_abs_path("Philipp_SiN.csv")
    elif material == "MgF2":
        pth = get_abs_path("Rodriguez-de Marcos_MgF2.csv")
        n_end = 960
        k_start = n_end + 3
    elif material == "CaF2":
        pth = get_abs_path("Daimon_CaF2.csv")
    elif material == "LiF":
        pth = get_abs_path("Li_LiF.csv")
    elif material == "Ta2O5":
        pth = get_abs_path("Gao_Ta2O5.csv")
        n_end = 726
        k_start = n_end + 3
    elif material == "TiO2":
        pth = get_abs_path("Sarkar_TiO2.csv")
        n_end = 977
        k_start = n_end + 3
    elif material == "Nb2O5":
        pth = get_abs_path("Lemarchand_Nb2O5.csv")
        n_end = 451
        k_start = n_end + 3
    elif material == "Silica":
        pth = get_abs_path("Malitson_Silica.csv")

    # bunch of conditionals on if we have extinction coefficients
    if n_end is not None:
        n_data = np.genfromtxt(pth, skip_header=1, delimiter=",")[:n_end].T
        if k_start is not None:
            k_data = np.genfromtxt(pth, skip_header=1, delimiter=",")[k_start:].T

    else:
        n_data = np.genfromtxt(pth, skip_header=1, delimiter=",").T

    # create the index splines
    n_model = interp1d(n_data[0], n_data[1])
    if k_start is not None:
        k_model = interp1d(k_data[0], k_data[1])
        index_model = lambda wvl: n_model(wvl) + 1j * k_model(wvl)
    else:
        index_model = n_model

    return index_model
