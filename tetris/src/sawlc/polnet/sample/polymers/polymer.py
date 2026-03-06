from abc import ABC, abstractmethod

from polnet.utils.poly import *
from polnet.utils.affine import *

from .monomer import Monomer

class Polymer(ABC):
    """
    Abstract class for modeling a Polymer (a sequence of monomers)
    """

    def __init__(self, m_surf, id0=0, code0=""):
        """
        Constructor

        :param m_surf: monomer surface (as vtkPolyData object)
        :param id0: id for the initial monomer (default 0)
        :param code0: code string for the initial monomer (default '')
        """
        h_diam = poly_diam(m_surf)
        assert h_diam > 0
        self.__m_surf = m_surf
        self.__m_diam = h_diam
        self.__m = list()
        self.__p = None
        self.__u = None
        self.__t, self.__r, self.__q = list(), list(), list()
        self.__t_length = 0
        self.__ids = list()
        self.__codes = list()

    def get_num_mmers(self):
        """
        :return: the number of monomers
        """
        return len(self.__ids)

    def get_mmer_ids(self):
        """
        :return: the list of mmer ids
        """
        return self.__ids

    def get_mmer_id(self, m_id):
        """
        Get monomer id from its position

        :param m_id: monomoer position in the polymer
        :return: an integer with the monomer id
        """
        return self.__ids[m_id]

    def get_mmer_code(self, m_id):
        """
        Get monomer code from its position

        :param m_id: monomoer position in the polymer
        :return: an string with the monomer code
        """
        return self.__codes[m_id]

    def get_vol(self):
        """
        Get the polymer volume

        :return: the computed volume
        """
        vol = 0
        if len(self.__m) == 0:
            return vol
        else:
            for m in self.__m:
                vol += m.get_vol()
            return vol

    def get_area(self, mode="sphere"):
        """
        Get the polymer area projected in a surface

        :param mode: computations mode, valid: 'sphere' each monomer is approximated to a sphere
        :return: the computed volume
        """
        area = 0
        if len(self.__m) == 0:
            return area
        else:
            for m in self.__m:
                area += m.get_area()
            return area

    def get_total_len(self):
        return self.__t_length

    def get_num_monomers(self):
        return len(self.__m)

    def get_monomer(self, m_id):
        """
        Get a monomer

        :param m_id: monomer id, it must be [0, get_num_monomers()-1]
        :return: the Monomer instance
        """
        return self.__m[m_id]

    def get_mmers_list(self):
        """
        Get all polymer's monomers in a list
        """
        return self.__m

    def get_mmer_center(self, m_id):
        """
        :param m_id: monomner id
        :return: monomer coordinates center
        """
        return self.__r[m_id]

    def get_mmer_rotation(self, m_id):
        """
        :param m_id: monomner id
        :return: monomer rotation as a quaternion
        """
        return self.__q[m_id]

    def get_tail_point(self):
        """
        Get the central coordinate for the latest monomer

        :return: point coordinates as ndarray
        """
        return self.__r[-1]

    def get_vtp(self):
        """
        Get the polymer a skeleton, each momomer is point and lines conecting monomers

        :return: a vtkPolyData
        """

        app_flt = vtk.vtkAppendPolyData()

        # Polymers loop
        for m_id in range(len(self.__m)):
            m_poly = self.__m[m_id].get_vtp()
            add_label_to_poly(m_poly, self.__ids[m_id], GTRUTH_VTP_LBLS)
            app_flt.AddInputData(m_poly)
        app_flt.Update()

        return app_flt.GetOutput()

    def get_skel(self, add_verts=True, add_lines=True, verts_rad=0):
        """
        Get the polymer as a skeleton, each monomer is a point or sphere and lines connecting monomers

        :param add_verts: if True (default) the vertices are included in the vtkPolyData
        :param add_lines: if True (default) the lines are included in the vtkPolyData
        :param verts_rad: if verts is True then sets the vertex radius, if <=0 a vertices are just points
        :return: a vtkPolyData
        """

        # Initialization
        poly, points = vtk.vtkPolyData(), vtk.vtkPoints()
        verts, lines = vtk.vtkCellArray(), vtk.vtkCellArray()
        sph_points = list()

        # Monomers loop
        if len(self.__r) == 1:
            sph_points.append(self.__r[0])
            id_p0 = points.InsertNextPoint(self.__r[0])
            if add_verts and (verts_rad <= 0):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(id_p0)
        else:
            for i in range(1, len(self.__r)):
                sph_points.append(self.__r[i])
                id_p0, id_p1 = points.InsertNextPoint(
                    self.__r[i - 1]
                ), points.InsertNextPoint(self.__r[i])
                if add_verts and (verts_rad <= 0):
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(id_p0)
                if add_lines:
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(id_p0)
                    lines.InsertCellPoint(id_p1)

        # Construct poly
        poly.SetPoints(points)
        if add_verts and (verts_rad <= 0):
            poly.SetVerts(verts)
        if add_lines:
            poly.SetLines(lines)

        # Spheres case
        if add_verts and (verts_rad > 0):
            sph_vtp = points_to_poly_spheres(sph_points, verts_rad)
            poly = merge_polys(sph_vtp, poly)

        return poly

    def add_monomer(self, r, t, q, m, id=0, code=""):
        """
        Add a new monomer surface to the polymer once affine transformation is known

        :param r: center point
        :param t: tangent vector
        :param q: unit quaternion for rotation
        :param m: monomer
        :param id: monomer id (default 0), necessary to identify the monomer within a list in an intercalated polymer
        :param code: monomer code string (default '')
        :return:
        """
        assert (
            isinstance(r, np.ndarray)
            and isinstance(t, np.ndarray)
            and isinstance(q, np.ndarray)
            and isinstance(m, Monomer)
        )
        assert (len(r) == 3) and (len(t) == 3) and (len(q) == 4)
        self.__r.append(r)
        self.__t.append(t)
        self.__q.append(q)
        self.__m.append(m)
        self.__ids.append(id)
        self.__codes.append(code)
        # Update total length
        if self.get_num_monomers() <= 1:
            self.__t_length = 0
        else:
            self.__t_length += points_distance(self.__r[-1], self.__r[-2])

    def insert_density_svol(
        self, m_svol, tomo, v_size=1, merge="max", off_svol=None
    ):
        """
        Insert a polymer as set of subvolumes into a tomogram

        :param m_svol: input monomer (or list) sub-volume reference
        :param tomo: tomogram where m_svol is added
        :param v_size: tomogram voxel size (default 1)
        :param merge: merging mode, valid: 'min' (default), 'max', 'sum' and 'insert'
        :param off_svol: offset coordinates for sub-volume monomer center coordinates
        :return:
        """
        if isinstance(m_svol, np.ndarray):
            m_svol = [
                m_svol,
            ]
        for mmer, id in zip(self.__m, self.__ids):
            mmer.insert_density_svol(
                m_svol[id], tomo, v_size, merge=merge, off_svol=off_svol
            )

    @abstractmethod
    def set_reference(self):
        raise NotImplementedError

    @abstractmethod
    def gen_new_monomer(self):
        raise NotImplementedError

    def overlap_polymer(self, monomer, over_tolerance=0):
        """
        Determines if a monomer overlaps with other polymer's monomers

        :param monomer: input monomer
        :param over_tolerance: fraction of overlapping tolerance (default 0)
        :return: True if there is an overlapping and False otherwise
        """

        # Initialization
        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetTolerance(VTK_RAY_TOLERANCE)
        selector.Initialize(monomer.get_vtp())

        # Polymer loop, no need to process monomer beyond diameter distance
        diam, center = monomer.get_diameter(), monomer.get_center_mass()
        for i in range(len(self.__m) - 1, 0, -1):
            hold_monomer = self.__m[i]
            dist = points_distance(center, hold_monomer.get_center_mass())
            if dist <= diam:
                poly_b = hold_monomer.get_vtp()
                count, n_points = 0.0, poly_b.GetNumberOfPoints()
                n_points_if = 1.0 / float(n_points)
                for i in range(n_points):
                    if selector.IsInsideSurface(poly_b.GetPoint(i)) > 0:
                        count += 1
                        over = count * n_points_if
                        if over > over_tolerance:
                            return True

        return False
