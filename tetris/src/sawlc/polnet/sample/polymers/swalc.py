

from .monomer import Monomer
from .polymer import Polymer
from polnet.utils.affine import *
import polnet.utils.poly as pp

class SAWLC(Polymer):
    """
    Class for fibers following model Self-Avoiding Worm-Like Chain (SAWLC)
    """

    def __init__(
        self, l_length, m_surf, p0=(0, 0, 0), id0=0, code0="", rot=None
    ):
        """
        Constructor

        :param l_lengh: link length
        :param m_surf: monomer surface (as vtkPolyData object)
        :param p0: starting point
        :param id0: id for the initial monomer
        :param code0: code string for the initial monomer (default '')
        :param rot: None by default, otherwise allow to externally determine the rotation
        """
        super(SAWLC, self).__init__(m_surf)
        assert l_length > 0
        self.__l = l_length
        self.set_reference(p0, id0=id0, code0=code0, rot=rot)

    def set_reference(self, p0=(0.0, 0.0, 0), id0=0, code0="", rot=None):
        """
        Initializes the chain with the specified point input point, if points were introduced before they are forgotten

        :param p0: starting point
        :param id0: id for the initial monomer
        :param code0: code string for the initial monomer (default '')
        :param rot: None by default, otherwise allow to extenerally determine the rotation
        :return:
        """
        assert hasattr(p0, "__len__") and (len(p0) == 3)
        self._Polymer__p = np.asarray(p0)
        hold_monomer = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)
        if rot is None:
            hold_q = gen_rand_unit_quaternion()
        else:
            assert hasattr(rot, "__len__") and (len(rot) == 4)
            hold_q = rot
        # hold_q = np.asarray((1, 0., 0., 1.), dtype=np.float32)
        hold_monomer.rotate_q(hold_q)
        hold_monomer.translate(p0)
        self.add_monomer(
            p0,
            np.asarray((0.0, 0.0, 0.0)),
            hold_q,
            hold_monomer,
            id=id0,
            code=code0,
        )

    def gen_new_monomer(
        self,
        over_tolerance=0,
        voi=None,
        v_size=1,
        fix_dst=None,
        ext_surf=None,
        rot=None,
    ):
        """
        Generates a new monomer for the polymer according to the specified random model

        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0)
        :param voi: VOI to define forbidden regions (default None, not applied)
        :param v_size: VOI voxel size, it must be greater than 0 (default 1)
        :param fix_dst: allows to set the distance for the new monomer externally (default None)
        :param ext_surf: allows to set the new mmer surface externally (default None)
        :param rot: None by default, otherwise allow to externally determine the rotation
        :return: a 4-tuple with monomer center point, associated tangent vector, rotated quaternion and monomer,
                 return None in case the generation has failed
        """

        # Translation
        if fix_dst is None:
            hold_l = self.__l
        else:
            hold_l = fix_dst
        t = gen_uni_s2_sample(np.asarray((0.0, 0.0, 0.0)), hold_l)
        r = self._Polymer__r[-1] + t

        # Rotation
        if rot is None:
            q = gen_rand_unit_quaternion()
        else:
            q = rot
        # q = np.asarray((1, 0, 0, 1), dtype=np.float32)

        # Monomer
        if ext_surf is None:
            hold_m = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)
        else:
            hold_m = Monomer(ext_surf, self._Polymer__m_diam)
        hold_m.rotate_q(q)
        hold_m.translate(r)

        # Check self-avoiding and forbidden regions
        if self.overlap_polymer(hold_m, over_tolerance=over_tolerance):
            return None
        elif voi is not None:
            if hold_m.overlap_voi(voi, v_size, over_tolerance=over_tolerance):
                return None

        return r, t, q, hold_m


class SAWLCPoly(Polymer):
    """
    Class for fibers following model Self-Avoiding Worm-Like Chain (SAWLC) on a PolyData
    """

    def __init__(
        self, poly, l_length, m_surf, p0=(0, 0, 0), id0=0, code="", rot=None
    ):
        """
        Constructor

        :param poly: vtkPolyData where the monomer center will be embedded
        :param l_lengh: link length
        :param m_surf: monomer surface (as vtkPolyData object)
        :param p0: starting point
        :param id0: id for the initial monomer (default 0)
        :param code0: code string for the initial monomer (default '')
        :param rot: None by default, otherwise allow to externally determine the rotation
        """
        super(SAWLCPoly, self).__init__(m_surf)
        assert isinstance(poly, vtk.vtkPolyData)
        assert l_length > 0
        self.__l = l_length
        self.__poly = poly
        self.set_reference(p0, id0=id0, code0=code, rot=rot)

    def set_reference(self, p0=(0.0, 0.0, 0), id0=0, code0="", rot=None):
        """
        Initializes the chain with the specified point input point, if points were introduced before they are forgotten

        :param p0: starting point
        :param id0: id for the initial monomer (default 0)
        :param code0: code string for the initial monomer (default '')
        :param rot: None by default, otherwise allow to externally determine the rotation
        :return:
        """
        assert hasattr(p0, "__len__") and (len(p0) == 3)
        self._Polymer__p, hold_n = pp.find_point_on_poly(
            np.asarray(p0), self.__poly
        )
        hold_monomer = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)
        # hold_q = np.asarray((1, 0., 0., 1.), dtype=np.float32)
        if rot is None:
            hold_q = pp.gen_rand_quaternion_on_vector(hold_n)
        else:
            assert hasattr(rot, "__len__") and (len(rot) == 4)
            hold_q = rot
        hold_monomer.rotate_q(hold_q)
        hold_monomer.translate(self._Polymer__p)
        self.add_monomer(
            self._Polymer__p,
            np.asarray((0.0, 0.0, 0.0)),
            hold_q,
            hold_monomer,
            id=id0,
            code=code0,
        )

    def gen_new_monomer(
        self,
        over_tolerance=0,
        voi=None,
        v_size=1,
        fix_dst=None,
        ext_surf=None,
        rot=None,
    ):
        """
        Generates a new monomer for the polymer according to the specified random model

        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0)
        :param voi: VOI to define forbidden regions (default None, not applied)
        :param v_size: VOI voxel size, it must be greater than 0 (default 1)
        :param fix_dst: allows to set the distance for the new monomer externally (default None)
        :param ext_surf: allows to set the new mmer surface externally (default None)
        :param rot: None by default, otherwise allow to extenerally determine the rotation
        :return: a 4-tuple with monomer center point, associated tangent vector, rotated quaternion and monomer,
                 return None in case the generation has failed
        """

        # Translation
        if fix_dst is None:
            hold_l = self.__l
        else:
            hold_l = fix_dst
        r = pp.gen_uni_s2_sample_on_poly(
            self._Polymer__r[-1], hold_l, 2, self.__poly
        )
        if r is None:
            return None
        r = np.asarray(r)
        t = r - self._Polymer__r[-1]

        # Rotation
        if rot is None:
            hold_n = pp.find_point_on_poly(r, self.__poly)[1]
            q = pp.gen_rand_quaternion_on_vector(hold_n)
            # q = np.asarray((1, 0, 0, 1), dtype=np.float32)
        else:
            q = rot

        # Monomer
        if ext_surf is None:
            hold_m = Monomer(self._Polymer__m_surf, self._Polymer__m_diam)
        else:
            hold_m = Monomer(ext_surf, self._Polymer__m_diam)
        hold_m.rotate_q(q)
        hold_m.translate(r)

        # Check self-avoiding and forbidden regions
        if self.overlap_polymer(hold_m, over_tolerance=over_tolerance):
            return None
        elif voi is not None:
            if hold_m.overlap_voi(voi, v_size, over_tolerance=over_tolerance):
                return None

        return r, t, q, hold_m