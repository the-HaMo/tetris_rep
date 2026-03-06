"""Classes for generating a membrane with Spherical shape

MbSphere: class for generating a membrane with Spherical shape
SphGen: Spherical memrbrane generator class
"""

import math
import random

import numpy as np
import scipy as sp

from .mb import Mb, MbGen
from .mb_factory import MbFactory
from polnet.utils.affine import lin_map
from polnet.utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbSphere(Mb):
    """Class for generating a membrane with Spherical shape"""

    def __init__(
        self,
        voi_shape: tuple,
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
        center: tuple[float, float, float] = (0, 0, 0),
        rad: float = 1,
    ) -> None:
        """Constructor

        Defines the basic properties of a spherical membrane and generates it.

        Args:
            voi_shape (tuple): reference volume of interest shape (X, Y and Z dimensions)
            v_size (float, optional): reference volume of interest voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.
            center (tuple, optional): center of the sphere in angstroms. Defaults to (0, 0, 0).
            rad (float, optional): radius of the sphere in angstroms. Defaults to 1.

        Raises:
            TypeError: if 'voi_shape' is not a tuple of three integers
            ValueError: if any dimension of 'voi_shape' is not an integer
            ValueError: if 'v_size' or 'thick' are not positive floats or 'layer_s' is negative
            TypeError: if 'center' is not a tuple of three floats
            ValueError: if any dimension of 'center' is not a float
            TypeError: if 'rad' is not a float
            ValueError: if 'rad' is not positive

        Returns:
            None
        """
        super(MbSphere, self).__init__(voi_shape, v_size, thick, layer_s)

        if not hasattr(center, "__len__") or (len(center) != 3):
            raise TypeError(
                "center must be a tuple of three floats (X, Y and Z)"
            )
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError("All dimensions of center must be floats")
        if not isinstance(rad, (int, float)):
            raise TypeError("rad must be a float")
        if rad <= 0:
            raise ValueError("rad must be positive")

        self.__center = np.array([float(c) for c in center])
        self.__rad = float(rad)
        self._Mb__build()

    def _Mb__build(self):
        """Generates the density, mask and surface of the spherical membrane

        Args:
            None

        Returns:
            None
        """

        # Input parsing
        t_v, s_v = (
            0.5 * (self._Mb__thick / self._Mb__v_size),
            self._Mb__layer_s / self._Mb__v_size,
        )
        rad_v = self.__rad / self._Mb__v_size
        ao_v = rad_v + t_v
        ai_v = rad_v - t_v
        ao_v_p1 = ao_v + 1
        ao_v_m1 = ao_v - 1
        ai_v_p1 = ai_v + 1
        ai_v_m1 = ai_v - 1
        p0_v = self.__center / self._Mb__v_size

        # Generating the bilayer
        dx, dy, dz = (
            float(self._Mb__voi_shape[0]),
            float(self._Mb__voi_shape[1]),
            float(self._Mb__voi_shape[2]),
        )
        dx2, dy2, dz2 = (
            math.floor(0.5 * dx),
            math.floor(0.5 * dy),
            math.floor(0.5 * dz),
        )
        p0_v[0] -= dx2
        p0_v[1] -= dy2
        p0_v[2] -= dz2
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(
            np.arange(x_l, x_h),
            np.arange(y_l, y_h),
            np.arange(z_l, z_h),
            indexing="ij",
        )

        # Mask generation
        R_o = (
            ((X - p0_v[0]) / ao_v) ** 2
            + ((Y - p0_v[1]) / ao_v) ** 2
            + ((Z - p0_v[2]) / ao_v) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v) ** 2
            + ((Y - p0_v[1]) / ai_v) ** 2
            + ((Z - p0_v[2]) / ai_v) ** 2
        )

        self._Mb__mask = np.logical_and(R_i >= 1, R_o <= 1)

        # Surface generation
        R_i = (
            ((X - p0_v[0]) / rad_v) ** 2
            + ((Y - p0_v[1]) / rad_v) ** 2
            + ((Z - p0_v[2]) / rad_v) ** 2
        )
        self._Mb__surf = iso_surface(R_i, 1)
        add_sfield_to_poly(
            self._Mb__surf,
            self._Mb__mask,
            "mb_mask",
            dtype="int",
            interp="NN",
            mode="points",
        )
        self._Mb__surf = poly_threshold(
            self._Mb__surf, "mb_mask", mode="points", low_th=0.5
        )

        # Outer layer
        R_o = (
            ((X - p0_v[0]) / ao_v_p1) ** 2
            + ((Y - p0_v[1]) / ao_v_p1) ** 2
            + ((Z - p0_v[2]) / ao_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ao_v_m1) ** 2
            + ((Y - p0_v[1]) / ao_v_m1) ** 2
            + ((Z - p0_v[2]) / ao_v_m1) ** 2
        )
        G = np.logical_and(R_i >= 1, R_o <= 1)

        # Inner layer
        R_o = (
            ((X - p0_v[0]) / ai_v_p1) ** 2
            + ((Y - p0_v[1]) / ai_v_p1) ** 2
            + ((Z - p0_v[2]) / ai_v_p1) ** 2
        )
        R_i = (
            ((X - p0_v[0]) / ai_v_m1) ** 2
            + ((Y - p0_v[1]) / ai_v_m1) ** 2
            + ((Z - p0_v[2]) / ai_v_m1) ** 2
        )
        G += np.logical_and(R_i >= 1, R_o <= 1)

        # Smoothing and normalization
        self._Mb__density = lin_map(
            -1 * sp.ndimage.gaussian_filter(G.astype(float), s_v), ub=0, lb=1
        )

    def __str__(self):
        center_str = f"({self.__center[0]:.2f}, {self.__center[1]:.2f}, {self.__center[2]:.2f})"
        return f"Membrane(Sphere), thickness={self._Mb__thick:.3f} Å, layer sigma={self._Mb__layer_s:.3f} Å, center={center_str}, radius={self.__rad:.3f} Å"


@MbFactory.register("sphere")
class SphGen(MbGen):
    """
    Spherical memrbrane generator class
    """

    def __init__(
        self,
        thick_rg: tuple[float, float],
        layer_s_rg: tuple[float, float],
        occ_rg: tuple[float, float],
        over_tol: float,
        mb_den_cf_rg: tuple[float, float],
        min_rad: float,
        max_rad: float = None,
    ) -> None:
        """
        Constructor

        Args:
            thick_rg (tuple[float, float]): tuple with the min and max thickness values.
            layer_s_rg (tuple[float, float]): tuple with the min and max layer sigma values.
            occ_rg (tuple[float, float]): tuple with the min and max occupancy values.
            over_tol (float): overlap tolerance for the membrane set.
            mb_den_cf_rg (tuple[float, float]): tuple with the min and max membrane density contrast values.
            min_rad (float): minimum radius of the sphere in angstroms.
            max_rad (float, optional): maximum radius of the sphere in angstroms.

        Returns:
            None
        """

        super(SphGen, self).__init__(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
        )
        self.__min_radius, self.__max_radius = min_rad, max_rad

    @classmethod
    def from_params(cls, params: dict) -> MbGen:
        """
        Creates a SphGen object from a dictionary of parameters

        Args:
            params (dict): dictionary with the membrane parameters

        Returns:
            SphGen: SphGen object
        """

        thick_rg = params.get("MB_THICK_RG", (25.0, 35.0))
        layer_s_rg = params.get("MB_LAYER_S_RG", (0.5, 2.0))
        occ_rg = params.get("MB_OCC_RG", (0.001, 0.003))
        over_tol = params.get("MB_OVER_TOL", 0.0)
        mb_den_cf_rg = params.get("MB_DEN_CF_RG", (0.3, 0.5))
        min_rad = params.get("MB_MIN_RAD", 75.0)
        max_rad = params.get("MB_MAX_RAD", None)

        return cls(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
            min_rad=min_rad,
            max_rad=max_rad,
        )

    def generate(self, voi_shape: tuple, v_size: float) -> MbSphere:
        """
        Generates a spherical membrane with random parameters within the input volume of interest shape

        Args:
            voi_shape: shape of the volume of interest
            v_size: voxel size

        Returns:
            MbSphere: generated spherical membrane
        """
        if self.__max_radius is None:
            self.__max_radius = math.sqrt(3) * max(voi_shape) * v_size

        radius = random.uniform(self.__min_radius, self.__max_radius)
        center = np.asarray(
            (
                voi_shape[0] * v_size * random.random(),
                voi_shape[1] * v_size * random.random(),
                voi_shape[2] * v_size * random.random(),
            )
        )
        thick = random.uniform(
            self._MbGen__thick_rg[0], self._MbGen__thick_rg[1]
        )
        layer_s = random.uniform(
            self._MbGen__layer_s_rg[0], self._MbGen__layer_s_rg[1]
        )

        return MbSphere(
            voi_shape=voi_shape,
            v_size=v_size,
            thick=thick,
            layer_s=layer_s,
            center=center,
            rad=radius,
        )
