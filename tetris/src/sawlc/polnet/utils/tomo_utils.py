"""
Utility functions for tomograms
"""

import numpy as np
import math


def insert_svol_tomo(svol, tomo, sub_pt, merge="max"):
    """
    Insert the content of a subvolume to a tomogram

    :param svol: input subvolume (or subtomogram)
    :param tomo: input tomogram that is going to be modified
    :param sub_pt: subvolume center point in the input tomogram
    :param merge: merging mode, valid: 'max' (default), 'min', 'sum' and 'insert'
    :return:
    """

    # Initialization
    sub_shape = svol.shape
    nx, ny, nz = sub_shape[0], sub_shape[1], sub_shape[2]
    mx, my, mz = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    mx1, my1, mz1 = mx - 1, my - 1, mz - 1
    hl_x, hl_y, hl_z = int(nx * 0.5), int(ny * 0.5), int(nz * 0.5)
    x, y, z = (
        int(round(sub_pt[0])),
        int(round(sub_pt[1])),
        int(round(sub_pt[2])),
    )

    # Compute bounding restriction
    # off_l_x, off_l_y, off_l_z = x - hl_x + 1, y - hl_y + 1, z - hl_z + 1
    off_l_x, off_l_y, off_l_z = x - hl_x, y - hl_y, z - hl_z
    # off_h_x, off_h_y, off_h_z = x + hl_x + 1, y + hl_y + 1, z + hl_z + 1
    off_h_x, off_h_y, off_h_z = x + hl_x, y + hl_y, z + hl_z
    dif_l_x, dif_l_y, dif_l_z = 0, 0, 0
    dif_h_x, dif_h_y, dif_h_z = nx, ny, nz
    if off_l_x < 0:
        # dif_l_x = abs(off_l_x) - 1
        dif_l_x = abs(off_l_x)
        off_l_x = 0
    if off_l_y < 0:
        # dif_l_y = abs(off_l_y) - 1
        dif_l_y = abs(off_l_y)
        off_l_y = 0
    if off_l_z < 0:
        # dif_l_z = abs(off_l_z) - 1
        dif_l_z = abs(off_l_z)
        off_l_z = 0
    if off_h_x >= mx:
        dif_h_x = nx - off_h_x + mx1
        off_h_x = mx1
    if off_h_y >= my:
        dif_h_y = ny - off_h_y + my1
        off_h_y = my1
    if off_h_z >= mz:
        dif_h_z = nz - off_h_z + mz1
        off_h_z = mz1
    if off_l_x > off_h_x:
        off_h_x = off_l_x
    if off_l_y > off_h_y:
        off_h_y = off_l_y
    if off_l_z > off_h_z:
        off_h_z = off_l_z
    if dif_l_x > dif_h_x:
        dif_h_x = dif_l_x
    if dif_l_y > dif_h_y:
        dif_h_y = dif_l_y
    if dif_l_z > dif_h_z:
        dif_h_z = dif_l_z
    sz_svol = [dif_h_x - dif_l_x, dif_h_y - dif_l_y, dif_h_z - dif_l_z]
    sz_off = [off_h_x - off_l_x, off_h_y - off_l_y, off_h_z - off_l_z]
    if (sz_svol[0] > sz_off[0]) and (sz_svol[0] > 1):
        dif_h_x -= 1
    if (sz_svol[1] > sz_off[1]) and (sz_svol[1] > 1):
        dif_h_y -= 1
    if (sz_svol[2] > sz_off[2]) and (sz_svol[2] > 1):
        dif_h_z -= 1

    # Modify the input tomogram
    if merge == "insert":
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = svol[
            dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z
        ]
    elif merge == "sum":
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] += svol[
            dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z
        ]
    elif merge == "min":
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = np.minimum(
            svol[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z],
            tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z],
        )
    elif merge == "max":
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = np.maximum(
            svol[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z],
            tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z],
        )
    elif merge == "and":
        tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z] = (
            np.logical_and(
                svol[dif_l_x:dif_h_x, dif_l_y:dif_h_y, dif_l_z:dif_h_z],
                tomo[off_l_x:off_h_x, off_l_y:off_h_y, off_l_z:off_h_z],
            )
        )


def density_norm(tomo, mask=None, inv=True):
    """
    Tomogram density normalization (I(x,y,z)-mean) / std)

    :param tomo: input tomogram
    :param mask: if None (default) the whole tomogram is used for computing the statistics otherwise just the masked region
    :param inv: if True the values are inverted (default)
    :return:
    """

    # Input parsing
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=bool)

    # Inversion
    if inv:
        hold_tomo = -1.0 * tomo
    else:
        hold_tomo = tomo

    # Statistics
    stat_tomo = hold_tomo[mask > 0]
    mn, st = stat_tomo.mean(), stat_tomo.std()

    # Histogram equalization
    tomo_out = np.zeros(shape=tomo.shape, dtype=np.float32)
    if st > 0:
        tomo_out = (hold_tomo - mn) / st
    else:
        print("WARNING (density_norm): standard deviation=" + str(st))

    return tomo_out


def trilin_interp(x, y, z, tomogram):
    """
    Trilinear interpolation of the value of a coordinate point within a tomogram

    :param x: x input coordinate
    :param y: y input coordinate
    :param z: z input coordinate
    :param tomogram: input ndarray with the scalar field
    :return: the value interpolated
    """

    # Input parsing
    assert isinstance(tomogram, np.ndarray) and len(tomogram.shape) == 3
    xc = int(math.ceil(x))
    yc = int(math.ceil(y))
    zc = int(math.ceil(z))
    xf = int(math.floor(x))
    yf = int(math.floor(y))
    zf = int(math.floor(z))
    assert (
        (xc < tomogram.shape[0])
        and (yc < tomogram.shape[1])
        and (zc < tomogram.shape[2])
        and (xf >= 0)
        and (yf >= 0)
        and (zf >= 0)
    )

    # Get neigbourhood values
    v000 = float(tomogram[xf, yf, zf])
    v100 = float(tomogram[xc, yf, zf])
    v010 = float(tomogram[xf, yc, zf])
    v001 = float(tomogram[xf, yf, zc])
    v101 = float(tomogram[xc, yf, zc])
    v011 = float(tomogram[xf, yc, zc])
    v110 = float(tomogram[xc, yc, zf])
    v111 = float(tomogram[xc, yc, zc])

    # Coordinates correction
    xn = x - xf
    yn = y - yf
    zn = z - zf
    x1 = 1 - xn
    y1 = 1 - yn
    z1 = 1 - zn

    # Interpolation
    return (
        (v000 * x1 * y1 * z1)
        + (v100 * xn * y1 * z1)
        + (v010 * x1 * yn * z1)
        + (v001 * x1 * y1 * zn)
        + (v101 * xn * y1 * zn)
        + (v011 * x1 * yn * zn)
        + (v110 * xn * yn * z1)
        + (v111 * xn * yn * zn)
    )


def nn_iterp(x, y, z, tomogram):
    """
    Nearest neighbour interpolation of the value of a coordinate point within a tomogram

    :param x: x input coordinate
    :param y: y input coordinate
    :param z: z input coordinate
    :param tomogram: input ndarray with the scalar field
    :return: the value interpolated
    """

    # Input parsing
    assert isinstance(tomogram, np.ndarray) and len(tomogram.shape) == 3
    xc = int(math.ceil(x))
    yc = int(math.ceil(y))
    zc = int(math.ceil(z))
    xf = int(math.floor(x))
    yf = int(math.floor(y))
    zf = int(math.floor(z))
    assert (
        (xc < tomogram.shape[0])
        and (yc < tomogram.shape[1])
        and (zc < tomogram.shape[2])
        and (xf >= 0)
        and (yf >= 0)
        and (zf >= 0)
    )

    # Finding the closest voxel
    point = np.asarray((x, y, z))
    X, Y, Z = np.meshgrid(
        range(xf, xc + 1), range(yf, yc + 1), range(zf, zc + 1), indexing="ij"
    )
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    min_point = np.asarray((X[0], Y[0], Z[0]))
    hold = point - min_point
    min_dist = np.sqrt((hold * hold).sum())
    for i in range(1, len(X)):
        hold_point = np.asarray((X[i], Y[i], Z[i]))
        hold = point - hold_point
        hold_dist = np.sqrt((hold * hold).sum())
        if hold_dist < min_dist:
            min_point = hold_point
            min_dist = hold_dist

    # Interpolation
    return tomogram[min_point[0], min_point[1], min_point[2]]
