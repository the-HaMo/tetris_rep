import numpy as np


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

def vol_cube(vol, off=0):
    """
    Reshape a 3D volume for being cubic

    :param vol: input volume (ndarray)
    :param off: offset voxels (default 0)
    :return: a cubic volume with the info from the input, the cube dimension corresponds with the largest input
             dimension
    """
    assert isinstance(vol, np.ndarray)
    assert len(vol.shape) == 3
    assert off >= 0
    dim_max = np.argmax(vol.shape)
    cube_dim = vol.shape[dim_max]
    out_vol = np.zeros(shape=(cube_dim, cube_dim, cube_dim), dtype=vol.dtype)
    if dim_max == 0:
        off_ly, off_lz = (cube_dim - vol.shape[1])//2, (cube_dim - vol.shape[2])//2
        off_hy, off_hz = off_ly + vol.shape[1], off_lz + vol.shape[2]
        out_vol[:,off_ly:off_hy,off_lz:off_hz] = vol
    elif dim_max == 1:
        off_lx, off_lz = (cube_dim - vol.shape[0])//2, (cube_dim - vol.shape[2])//2
        off_hx, off_hz = off_lx + vol.shape[0], off_lz + vol.shape[2]
        out_vol[off_lx:off_hx,:,off_lz:off_hz] = vol
    else:
        off_lx, off_ly = (cube_dim - vol.shape[0])//2, (cube_dim - vol.shape[1])//2
        off_hx, off_hy = off_lx + vol.shape[0], off_ly + vol.shape[1]
        out_vol[off_lx:off_hx,off_lx:off_hx,:] = vol
    if off > 0:
        off_2 = off // 2
        off_vol = np.zeros(shape=(cube_dim+off, cube_dim+off, cube_dim+off), dtype=vol.dtype)
        off_vol[off_2:off_2+cube_dim, off_2:off_2+cube_dim, off_2:off_2+cube_dim] = out_vol
        out_vol = off_vol
    return out_vol