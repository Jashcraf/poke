import matplotlib.pyplot as plt
import matplotlib as mpl
from poke.poke_math import np


def jones_pupil(raybundle, which=-1):
    """plot the jones pupil

    Parameters
    ----------
    raybundle : poke.Rayfront
        Rayfront that holds the jones pupil you wish to plot
    which : int, optional
        Which index of the jones pupil list to plot, by default -1
    """

    x = raybundle.xData[0][0]
    y = raybundle.yData[0][0]
    Jmat = raybundle.jones_pupil[which]

    fig, axs = plt.subplots(figsize=[12, 5], nrows=2, ncols=4)
    plt.suptitle("Jones Pupil")
    for j in range(2):
        for k in range(2):
            ax = axs[j, k]
            ax.set_title("|J{j}{k}|".format(j=j, k=k))
            sca = ax.scatter(x, y, c=np.abs(Jmat[..., j, k]), cmap="inferno")
            fig.colorbar(sca, ax=ax)

            # turn off the ticks
            if j != 1:
                ax.xaxis.set_visible(False)
            if k != 0:
                ax.yaxis.set_visible(False)

    for j in range(2):
        for k in range(2):

            # Offset the p coefficient
            if j == 1:
                if k == 1:
                    offset = 0  # -np.pi
                else:
                    offset = 0
            else:
                offset = 0

            ax = axs[j, k + 2]
            ax.set_title(r"$\angle$" + "J{j}{k}".format(j=j, k=k))
            sca = ax.scatter(x, y, c=np.angle(Jmat[..., j, k]) + offset, cmap="coolwarm")
            fig.colorbar(sca, ax=ax)

            # turn off the ticks
            if j != 1:
                ax.xaxis.set_visible(False)

            ax.yaxis.set_visible(False)
    plt.show()


def ray_opd(raybundle, which=-1):
    """plot the OPD of the ray trace

    Parameters
    ----------
    raybundle : poke.Rayfront
        the Rayfront that holds the data you wish to plot
    which : int, optional
        Which index of the jones pupil list to plot, by default -1
    """

    x = raybundle.xData[0, 0]
    y = raybundle.yData[0, 0]
    opd = raybundle.opd[0, -1]
    opd -= np.mean(opd)

    plt.figure(figsize=[5, 5])
    plt.title("OPD for raybundle [m]")
    plt.scatter(x, y, c=opd, cmap="coolwarm")
    plt.colorbar()
    plt.show()


def mueller_pupil(M):
    fig, axs = plt.subplots(figsize=[12, 12], nrows=4, ncols=4)
    plt.suptitle("Mueller Pupil")
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.set_title("J{i}{j}".format(i=i, j=j))
            sca = ax.imshow(M[i, j, :, :])
            fig.colorbar(sca, ax=ax)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    plt.show()


def point_spread_matrix(PSM):
    from matplotlib.colors import LogNorm

    fig, axs = plt.subplots(figsize=[12, 12], nrows=4, ncols=4)
    plt.suptitle("Point-spread Matrix")
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.set_title("M{i}{j}".format(i=i, j=j))
            sca = ax.imshow(PSM[..., i, j], cmap="coolwarm")
            fig.colorbar(sca, ax=ax)

            if i != 3:
                ax.axes.xaxis.set_visible(False)
            if j != 0:
                ax.axes.yaxis.set_visible(False)
    plt.show()
