"""Classes for generating a membrane with Toroidal shape

MbToroid: class for generating a membrane with Toroidal shape
ToroidGen: Toroidal membrane generator class
"""

import math
import random

import numpy as np
import scipy as sp

from .mb import Mb, MbGen
from .mb_factory import MbFactory
from polnet.utils.affine import lin_map, tomo_rotate
from polnet.utils.distribution import gen_rand_unit_quaternion, gen_bounded_exp 
from polnet.utils.tomo_utils import density_norm
from polnet.utils.poly import iso_surface, add_sfield_to_poly, poly_threshold

class MbToroid(Mb):
    """Class for generating a membrane with Toroidal shape"""

    def __init__(
        self,
        voi_shape: tuple,
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
        center: tuple[float, float, float] = (0, 0, 0),
        maj_rad: float = 1,
        min_rad: float = 1,
        rot_q: tuple[float, float, float, float] = (1, 0, 0, 0)
    ) -> None:
        """Constructor

        Defines the basic properties of a toroidal membrane and generates it.

        Args:
            voi_shape (tuple): reference tomogram shape (X, Y and Z dimensions)
            v_size (float, optional): reference tomogram voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.
            center (tuple, optional): center of the toroid in angstroms. Defaults to (0, 0, 0).
            maj_rad (float, optional): major radius of the toroid in angstroms. Defaults to 1.
            min_rad (float, optional): minor radius of the toroid in angstroms. Defaults to 1.
            rot_q (tuple[float, float, float, float], optional): rotation quaternion of the toroid. Defaults to (1, 0, 0, 0).

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
        super(MbToroid, self).__init__(voi_shape, v_size, thick, layer_s)

        if not hasattr(center, "__len__") or (len(center) != 3):
            raise TypeError(
                "center must be a tuple of three floats (X, Y and Z)"
            )
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError("All dimensions of center must be floats")
        if not isinstance(maj_rad, (int, float)):
            raise TypeError("maj_rad must be a float")
        if maj_rad <= 0:
            raise ValueError("maj_rad must be positive")
        if not isinstance(min_rad, (int, float)):
            raise TypeError("min_rad must be a float")
        if min_rad <= 0:
            raise ValueError("min_rad must be positive")

        self.__center = np.array([float(c) for c in center])
        self.__maj_rad = float(maj_rad)
        self.__min_rad = float(min_rad)
        self.__rot_q = rot_q
        self._Mb__build()

    def _Mb__build(self):

        # Input parsing
        t_v, s_v = .5 * (self._Mb__thick / self._Mb__v_size), self._Mb__layer_s / self._Mb__v_size
        rad_a_v, rad_b_v = self.__maj_rad / self._Mb__v_size, self.__min_rad / self._Mb__v_size
        bo_v, bi_v = rad_b_v + t_v, rad_b_v - t_v
        bo_v_p1, bo_v_m1 = bo_v + 1, bo_v - 1
        bi_v_p1, bi_v_m1 = bi_v + 1, bi_v - 1
        p0_v = self.__center / self._Mb__v_size

        # Generating the bilayer
        dx, dy, dz = float(self._Mb__voi_shape[0]), float(self._Mb__voi_shape[1]), float(self._Mb__voi_shape[2])
        dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
        p0_v[0] -= dx2
        p0_v[1] -= dy2
        p0_v[2] -= dz2
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')

        # Mask generation
        R_o = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bo_v*bo_v) <= 1
        R_i = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bi_v*bi_v) >= 1
        self._Mb__mask = tomo_rotate(np.logical_and(R_i, R_o), self.__rot_q, order=0)

        # Surface generation
        R_i = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - rad_b_v*rad_b_v)
        R_i = tomo_rotate(R_i, self.__rot_q, mode='reflect')
        self._Mb__surf = iso_surface(R_i, 1)
        add_sfield_to_poly(self._Mb__surf, self._Mb__mask, 'mb_mask', dtype='int', interp='NN', mode='points')
        self._Mb__surf = poly_threshold(self._Mb__surf, 'mb_mask', mode='points', low_th=.5)

        # Outer layer
        R_o = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bo_v_p1*bo_v_p1) <= 1
        R_i = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bo_v_m1*bo_v_m1) >= 1
        G = tomo_rotate(np.logical_and(R_i, R_o), self.__rot_q, order=0)

        # Inner layer
        R_o = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bi_v_p1 * bi_v_p1) <= 1
        R_i = ((rad_a_v - np.sqrt((X-p0_v[0])**2 + (Y-p0_v[1])**2))**2 + (Z-p0_v[2])**2 - bi_v_m1 * bi_v_m1) >= 1
        G += tomo_rotate(np.logical_and(R_i, R_o), self.__rot_q, order=0)

        # Smoothing
        self._Mb__density = lin_map(density_norm(sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True), ub=0, lb=1)

    def __str__(self):
        center_str = f"({self.__center[0]:.2f}, {self.__center[1]:.2f}, {self.__center[2]:.2f})"
        maj_rad_str = f"{self.__maj_rad:.2f}"
        min_rad_str = f"{self.__min_rad:.2f}"
        return f"Membrane(Toroid), thickness={self._Mb__thick:.3f} Å, layer sigma={self._Mb__layer_s:.3f} Å, center={center_str}, major radius={maj_rad_str} Å, minor radius={min_rad_str} Å"


@MbFactory.register("toroid")
class TorGen(MbGen):
    """
    Toroidal membrane generator class
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
            min_rad (float): minimum radius of the toroid (both radii) in angstroms.
            max_rad (float, optional): maximum radius of the toroid (both radii)in angstroms.

        Returns:
            None
        """

        super(TorGen, self).__init__(
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
        Creates a ToroidGen object from a dictionary of parameters

        Args:
            params (dict): dictionary with the membrane parameters

        Returns:
            TorGen: ToroidGen object
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

    def generate(self, voi_shape: tuple, v_size: float) -> MbToroid:
        """
        Generates a toroidal membrane with random parameters within the input volume of interest shape

        Args:
            voi_shape: shape of the volume of interest
            v_size: voxel size

        Returns:
            MbToroid: generated toroidal membrane
        """
        if self.__max_radius is None:
            self.__max_radius = math.sqrt(3) * max(voi_shape) * v_size
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
        major_radius = random.uniform(self.__min_radius, self.__max_radius)
        minor_radius = random.uniform(self.__min_radius, self.__max_radius)
        if minor_radius >= major_radius:
            swap = minor_radius
            minor_radius = major_radius
            major_radius = swap

        return MbToroid(
            voi_shape=voi_shape,
            v_size=v_size,
            thick=thick,
            layer_s=layer_s,
            center=center,
            maj_rad=major_radius,
            min_rad=minor_radius,
            rot_q=gen_rand_unit_quaternion()
        )
