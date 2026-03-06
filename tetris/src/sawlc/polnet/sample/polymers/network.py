from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import vtk

import polnet.utils.poly as pp
from polnet.utils.affine import tomo_rotate
from polnet.utils.tomo_utils import insert_svol_tomo

NET_TYPE_STR = "net_type"


class Network(ABC):
    """
    General class for a network of polymers
    """

    def __init__(self, voi, v_size, svol=None):
        """
        Construction

        :param voi: a 3D numpy array to define a VOI (Volume Of Interest) for polymers
        :param v_size: voxel size (default 1)
        :param svol: monomer subvolume (or list of) as a numpy ndarray (default None)
        :param mb_area: total membrane area within the same VOI as the network (deftault None)
        """
        self.set_voi(voi)
        self.__vol = (
            float((self.__voi > 0).sum()) * v_size * v_size * v_size
        )  # withou the float cast is my raise overflow warning in Windows
        self.__v_size = v_size
        self.__pl_occ = 0
        self.__pl = list()
        self.__pl_nmmers = list()
        self.__svol = svol
        if self.__svol is not None:
            if not hasattr(svol, "__len__"):
                assert isinstance(self.__svol, np.ndarray)
        self.__min_nmmer = 1
        self.__poly_area = 0
        self.__pmer_fails = 0

    def set_min_nmmer(self, min_nmmer):
        """
        Set a minimum number of monomers for the generated filaments

        :param min_nmmer: integer with the minimum number of monomenrs per filament
        :return:
        """
        self.__min_nmmer = int(min_nmmer)

    def get_pmer_fails(self):
        return self.__pmer_fails

    def get_pmers_list(self):
        return self.__pl

    def get_num_pmers(self):
        """
        :return: the number of polymers in the network
        """
        return len(self.__pl)

    def get_num_mmers(self):
        """
        :return: the number of monomers in the network
        """
        count_mmers = 0
        for pl in self.__pl:
            count_mmers += pl.get_num_mmers()
        return count_mmers

    def get_polymer_occupancy(self):
        return self.__pl_occ

    def add_polymer(self, polymer, occ_mode="volume"):
        """
        Add a new polymer to the network

        :param polymer: polymer to add
        :param occ_mode: occupancy mode, valid: 'volume' (default), 'area' for membrane-bound polymer
        :return:
        """
        assert (occ_mode == "volume") or (occ_mode == "area")
        self.__pl.append(polymer)
        self.__pl_nmmers.append(polymer.get_num_mmers())
        if occ_mode == "volume":
            self.__pl_occ += 100.0 * (polymer.get_vol() / self.__vol)
        else:
            self.__pl_occ += 100.0 * (
                polymer.get_area() / self._Network__poly_area
            )
        # print('Occ: ', self.__pl_occ)

    @abstractmethod
    def build_network(self):
        """
        Builds an instance of the network

        :return: None"""
        raise NotImplemented

    def get_voi(self):
        """
        Get the VOI

        :return: an ndarray
        """
        return self.__voi

    def get_gtruth(self, thick=1):
        """
        Get the ground truth tomogram

        :param thick: ground truth tickness in voxels (default 1)
        :return: a binary numpy 3D array
        """
        hold_gtruth = self.gen_vtp_points_tomo()
        if thick >= 1:
            hold_gtruth = sp.ndimage.morphology.binary_dilation(
                hold_gtruth, iterations=int(thick)
            )
        return hold_gtruth

    def set_voi(self, voi):
        """
        Set the VOI

        :param voi:
        """
        assert isinstance(voi, np.ndarray)
        if voi.dtype is bool:
            self.__voi = voi
        else:
            self.__voi = voi > 0

    def get_vtp(self):
        """
        Get Polymers Network as a vtkPolyData with their surfaces

        :return: a vtkPolyData
        """

        app_flt = vtk.vtkAppendPolyData()

        # Polymers loop
        for pol in self.__pl:
            app_flt.AddInputData(pol.get_vtp())
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
        app_flt = vtk.vtkAppendPolyData()

        # Polymers loop
        for pol in self.__pl:
            app_flt.AddInputData(pol.get_skel(add_verts, add_lines, verts_rad))

        # Update and return
        app_flt.Update()
        return app_flt.GetOutput()

    def gen_vtp_points_tomo(self):
        """
        Generates a binary tomogram where True elements correspond with the polydata closes voxel projection

        :return: a binary VOI shaped numpy array
        """
        nx, ny, nz = self.__voi.shape
        hold_tomo = np.zeros(shape=(nx, ny, nz), dtype=bool)
        hold_vtp_skel = self.get_skel()
        for i in range(hold_vtp_skel.GetNumberOfPoints()):
            x, y, z = hold_vtp_skel.GetPoint(i)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            if (
                (x >= 0)
                and (y >= 0)
                and (z >= 0)
                and (x < nx)
                and (y < ny)
                and (z < nz)
            ):
                hold_tomo[x, y, z] = True
        return hold_tomo

    def insert_density_svol(
        self, m_svol, tomo, v_size=1, merge="max", off_svol=None
    ):
        """
        Insert a polymer network as set of subvolumes into a tomogram

        :param m_svol: input monomer (or list) sub-volume reference
        :param tomo: tomogram where m_svol is added
        :param v_size: tomogram voxel size (default 1)
        :param merge: merging mode, valid: 'min' (default), 'max', 'sum' and 'insert'
        :param off_svol: offset coordinates for sub-volume monomer center coordinates
        """
        if not hasattr(m_svol, "__len__"):
            assert isinstance(m_svol, np.ndarray) and (len(m_svol.shape) == 3)
        assert isinstance(tomo, np.ndarray) and (len(tomo.shape) == 3)
        assert (
            (merge == "max")
            or (merge == "min")
            or (merge == "sum")
            or (merge == "insert")
        )
        assert v_size > 0
        if off_svol is not None:
            assert isinstance(off_svol, np.ndarray) and (len(off_svol) == 3)

        for pl in self.__pl:
            pl.insert_density_svol(
                m_svol, tomo, v_size, merge=merge, off_svol=off_svol
            )

    def add_monomer_to_voi(self, mmer, mmer_svol=None):
        """
        Adds a monomer to VOI mask

        :param mmer: monomer to define rigid transformations
        :param mmer_voi: subvolume (binary numpy ndarray) with monomer VOI
        """
        assert isinstance(mmer_svol, np.ndarray) and (mmer_svol.dtype == bool)
        v_size_i = 1.0 / self.__v_size
        tot_v = np.asarray((0.0, 0.0, 0.0))
        hold_svol = mmer_svol > 0
        for trans in mmer.get_trans_list():
            if trans[0] == "t":
                tot_v += trans[1] * v_size_i
            elif trans[0] == "r":
                hold_svol = tomo_rotate(
                    hold_svol,
                    trans[1],
                    order=0,
                    mode="constant",
                    cval=hold_svol.max(),
                    prefilter=False,
                )
                # hold_svol = tomo_rotate(hold_svol, trans[1], mode='constant', cval=hold_svol.min())
        insert_svol_tomo(hold_svol, self.__voi, tot_v, merge="min")

    def count_proteins(self):
        """
        Genrrates output statistics for this network

        :return: a dictionary with the number of proteins for protein id
        """
        counts = dict()
        for pl in self.__pl:
            ids = pl.get_mmer_ids()
            for id in ids:
                try:
                    counts[id] += 1
                except KeyError:
                    counts[id] = 0
        return counts