import numpy as np
import vtk

from polnet.utils.poly import *
from polnet.utils.affine import poly_rotate_wxyz, quat_to_angle_axis, poly_translate, tomo_rotate, vect_rotate, points_distance
from polnet.utils.tomo_utils import insert_svol_tomo

MB_DOMAIN_FIELD_STR = "mb_domain"

class Monomer:
    """
    Class for a single monomer
    """

    def __init__(self, m_surf, diam):
        """
        Constructor

        :param m_surf: monomer surface (as vtkPolyData object)
        :param diam: monomer diameter
        """
        assert isinstance(m_surf, vtk.vtkPolyData)
        assert diam > 0
        self.__m_surf = vtk.vtkPolyData()
        self.__m_surf.DeepCopy(m_surf)
        self.__diam = diam
        self.__rot_angs = np.asarray((0.0, 0.0, 0.0))
        self.__bounds = np.zeros(shape=6)
        self.compute_bounds()
        # Ordered transformation queue, each entry is a 2-tuple
        # (str in ['t', 'r'], transform value in [vector, quaternion])
        self.__trans = list()

    def get_vtp(self):
        return self.__m_surf

    def get_center_mass(self):
        """
        Computer and return the monomer center of mass

        :return: a numpy array
        """
        return np.asarray(poly_center_mass(self.__m_surf))

    def get_diameter(self):
        return self.__diam

    def get_trans_list(self):
        """
        Get transformations list

        :return: a list with al transformations, each element is duple with a first element
        indicating the transformation type ('r' or 't')
        """
        return self.__trans

    def compute_bounds(self):
        # Compute bounds
        arr = self.__m_surf.GetPoints().GetData()
        self.__bounds[0], self.__bounds[1] = arr.GetRange(0)
        self.__bounds[2], self.__bounds[3] = arr.GetRange(1)
        self.__bounds[4], self.__bounds[5] = arr.GetRange(2)

    def rotate_q(self, q):
        """
        Applies rotation rigid transformation around center from an input unit quaternion.

        :param q: input quaternion
        :return:
        """
        w, v_axis = quat_to_angle_axis(q[0], q[1], q[2], q[3])
        self.__m_surf = poly_rotate_wxyz(
            self.__m_surf, w, v_axis[0], v_axis[1], v_axis[2]
        )
        self.compute_bounds()
        self.__trans.append(("r", q))

    def translate(self, t_v):
        """
        Applies rotation rigid transformation.

        :param t_v: translation vector (x, y, z)
        :return:
        """
        self.__m_surf = poly_translate(self.__m_surf, t_v)
        self.compute_bounds()
        self.__trans.append(("t", t_v))

    def point_in_bounds(self, point):
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > point[0]) or (self.__bounds[1] < point[0]):
            x_over = False
        if (self.__bounds[2] > point[1]) or (self.__bounds[3] < point[1]):
            y_over = False
        if (self.__bounds[4] > point[2]) or (self.__bounds[5] < point[2]):
            y_over = False
        return x_over and y_over and z_over

    def bound_in_bounds(self, bounds):
        """
        Check if the object's bound are at least partially in another bound

        :param bounds: input bound
        :return:
        """
        x_over, y_over, z_over = True, True, True
        if (self.__bounds[0] > bounds[1]) or (self.__bounds[1] < bounds[0]):
            x_over = False
        if (self.__bounds[2] > bounds[3]) or (self.__bounds[3] < bounds[2]):
            y_over = False
        if (self.__bounds[4] > bounds[5]) or (self.__bounds[5] < bounds[4]):
            y_over = False
        return x_over and y_over and z_over

    def overlap_voi(self, voi, v_size=1, over_tolerance=0):
        """
        Determines if the monomer overlaps a VOI, that requires the next condition:
            - Any particle on the monomer surface is within the VOI

        :param voi: input VOI (Volume Of Interest), binary tomogram with True for VOI voxels
        :param v_size: voxel size, it must greater than 0 (default 1)
        :param over_tolerance: maximum overlap allowed (default 0)
        :return: True if the monomer overlaps the VOI, False otherwise
        """

        # Initialization
        assert v_size > 0
        assert isinstance(voi, np.ndarray) and (voi.dtype == "bool")
        nx, ny, nz = voi.shape
        v_size_i = 1.0 / v_size
        mbd_prop = self.__m_surf.GetPointData().GetArray(MB_DOMAIN_FIELD_STR)

        # Any particle on the monomer surface is within the VOI
        count, n_points = 0.0, self.__m_surf.GetNumberOfPoints()
        n_points_if = 1.0 / float(n_points)
        if mbd_prop is None:
            for i in range(self.__m_surf.GetNumberOfPoints()):
                pt = np.asarray(self.__m_surf.GetPoint(i)) * v_size_i
                x, y, z = np.round(pt).astype(int)
                if (
                    (x < nx)
                    and (y < ny)
                    and (z < nz)
                    and (x >= 0)
                    and (y >= 0)
                    and (z >= 0)
                ):
                    if not voi[x, y, z]:
                        count += 1
                        over = count * n_points_if
                        if over > over_tolerance:
                            return True
                else:
                    count += 1
                    over = count * n_points_if
                    if over > over_tolerance:
                        return True
        else:
            for i in range(self.__m_surf.GetNumberOfPoints()):
                if mbd_prop.GetValue(i) == 0:
                    pt = np.asarray(self.__m_surf.GetPoint(i)) * v_size_i
                    x, y, z = np.round(pt).astype(int)
                    if (
                        (x < nx)
                        and (y < ny)
                        and (z < nz)
                        and (x >= 0)
                        and (y >= 0)
                        and (z >= 0)
                    ):
                        if not voi[x, y, z]:
                            count += 1
                            over = count * n_points_if
                            if over > over_tolerance:
                                return True
                    else:
                        count += 1
                        over = count * n_points_if
                        if over > over_tolerance:
                            return True

        return False

    def get_vol(self):
        """
        Get the polymer volume
        :return: the computed volume
        """
        return poly_volume(self.__m_surf)

    def get_area(self):
        """
        Get the polymer area projected on a surface (currently only a plane containing the center, the monomer is
        approximated as ssphere)

        :return: the computed area
        """
        diam = poly_diam(self.__m_surf)
        return np.pi * diam * diam * 0.25

    def get_copy(self):
        """
        Get a copy of the current Monomer
        :return: a new instance of this monomer
        """
        return Monomer(self.__m_surf, self.__diam)

    def insert_density_svol(
        self, m_svol, tomo, v_size=1, merge="max", off_svol=None
    ):
        """
        Insert a monomer subvolume into a tomogram

        :param m_svol: input monomer sub-volume
        :param tomo: tomogram where m_svol is added
        :param v_size: tomogram voxel size (default 1)
        :param merge: merging mode, valid: 'min' (default), 'max', 'sum' and 'insert'
        :param off_svol: offset coordinates in voxels for shifting sub-volume monomer center coordinates (default None)
        :return:
        """
        v_size_i = 1.0 / v_size
        tot_v = np.asarray((0.0, 0.0, 0.0))
        hold_svol = m_svol
        for trans in self.__trans:
            if trans[0] == "t":
                tot_v += trans[1] * v_size_i
            elif trans[0] == "r":
                if merge == "min":
                    if hold_svol.dtype == bool:
                        hold_svol = tomo_rotate(
                            hold_svol,
                            trans[1],
                            order=0,
                            mode="constant",
                            cval=hold_svol.max(),
                        )
                    else:
                        hold_svol = tomo_rotate(
                            hold_svol,
                            trans[1],
                            mode="constant",
                            cval=hold_svol.max(),
                        )
                else:
                    if hold_svol.dtype == bool:
                        hold_svol = tomo_rotate(
                            hold_svol,
                            trans[1],
                            order=0,
                            mode="constant",
                            cval=hold_svol.min(),
                        )
                    else:
                        hold_svol = tomo_rotate(
                            hold_svol,
                            trans[1],
                            mode="constant",
                            cval=hold_svol.min(),
                        )
                if off_svol is not None:
                    off_svol = vect_rotate(off_svol, trans[1])
        if off_svol is not None:
            tot_v += off_svol
        insert_svol_tomo(hold_svol, tomo, tot_v, merge=merge)

    def overlap_mmer(self, mmer, over_tolerance=0):
        """
        Determines if the monomer overlaps with another

        :param mmer: input monomer to check overlap with self
        :param over_tolerance: maximum overlap allowed (default 0)
        :return: True if overlapping, otherwise False
        """
        # Initialization
        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetTolerance(VTK_RAY_TOLERANCE)
        selector.Initialize(self.get_vtp())

        dist = points_distance(self.get_center_mass(), mmer.get_center_mass())
        if dist <= self.get_diameter():
            poly_b = mmer.get_vtp()
            count, n_points = 0.0, poly_b.GetNumberOfPoints()
            n_points_if = 1.0 / float(n_points)
            for i in range(n_points):
                if selector.IsInsideSurface(poly_b.GetPoint(i)) > 0:
                    count += 1
                    over = count * n_points_if
                    if over > over_tolerance:
                        return True

        return False

    def overlap_net(self, net, over_tolerance=0, max_dist=None):
        """
        Determines if the monomer overlaps with another momonmer in a network

        :param mmer: input monomer to check overlap with self
        :param over_tolerance: maximum overlap allowed (default 0)
        :param max_dist: allows to externally set a maximum distance (in A) to seach for collisions, otherwise 1.2 monomer
                         diameter is used
        :return: True if overlapping, otherwise False
        """
        # Initialization
        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetTolerance(VTK_RAY_TOLERANCE)
        selector.Initialize(self.get_vtp())

        for pmer in net.get_pmers_list():
            for mmer in pmer.get_mmers_list():
                dist = points_distance(
                    self.get_center_mass(), mmer.get_center_mass()
                )
                if max_dist is None:
                    max_dist_h = self.get_diameter() * 1.2
                else:
                    max_dist_h = max_dist
                if dist <= max_dist_h:
                    poly_b = mmer.get_vtp()
                    count, n_points = 0.0, poly_b.GetNumberOfPoints()
                    n_points_if = 1.0 / float(n_points)
                    for i in range(n_points):
                        if selector.IsInsideSurface(poly_b.GetPoint(i)) > 0:
                            count += 1
                            over = count * n_points_if
                            if over > over_tolerance:
                                return True

        return False