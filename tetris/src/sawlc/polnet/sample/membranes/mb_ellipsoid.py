"""Classes for generating a membrane with Ellipsoidal shape

MbEllipsoid: class for generating a membrane with Ellipsoidal shape
EllipsoidGen: Ellipsoidal membrane generator class
"""

import math
import random

import numpy as np
import scipy as sp

from .mb import Mb, MbGen, MbError
from .mb_factory import MbFactory
from polnet.utils.affine import lin_map, tomo_rotate
from polnet.utils.distribution import gen_rand_unit_quaternion, gen_bounded_exp 
from polnet.utils.tomo_utils import density_norm
from polnet.utils.poly import iso_surface, add_sfield_to_poly, poly_threshold


class MbEllipsoid(Mb):
    """Class for generating a membrane with Ellipsoidal shape"""

    def __init__(
        self,
        voi_shape: tuple,
        v_size: float = 1,
        thick: float = 1,
        layer_s: float = 1,
        center: tuple[float, float, float] = (0, 0, 0),
        semi_axes: tuple[float, float, float] = (1, 1, 1),
        rot_q: np.ndarray = (1, 0, 0, 0)
    ) -> None:
        """Constructor

        Defines the basic properties of a spherical membrane and generates it.

        Args:
            voi_shape (tuple): reference tomogram shape (X, Y and Z dimensions)
            v_size (float, optional): reference tomogram voxel size in angstroms. Defaults to 1.
            thick (float, optional): membrane thickness in angstroms. Defaults to 1.
            layer_s (float, optional): Gaussian sigma for each layer in angstroms. Defaults to 1.
            center (tuple, optional): center of the ellipsoid in angstroms. Defaults to (0, 0, 0).
            semi_axes (tuple, optional): semi-axes of the ellipsoid in angstroms. Defaults to (1, 1, 1).
            rot_q (np.ndarray, optional): rotation quaternion. Defaults to (1, 0, 0, 0).

        Raises:
            TypeError: if 'voi_shape' is not a tuple of three integers
            ValueError: if any dimension of 'voi_shape' is not an integer
            ValueError: if 'v_size' or 'thick' are not positive floats or 'layer_s' is negative
            TypeError: if 'center' is not a tuple of three floats
            ValueError: if any dimension of 'center' is not a float
            MbError: if generated membrane has zero volume

        Returns:
            None
        """
        super(MbEllipsoid, self).__init__(voi_shape, v_size, thick, layer_s)

        if not hasattr(center, "__len__") or (len(center) != 3):
            raise TypeError(
                "center must be a tuple of three floats (X, Y and Z)"
            )
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError("All dimensions of center must be floats")
        if not hasattr(semi_axes, "__len__") or (len(semi_axes) != 3):
            raise TypeError(
                "semi_axes must be a tuple of three floats (a, b and c)"
            )
        if not all(isinstance(sa, (int, float)) for sa in semi_axes):
            raise TypeError("All dimensions of semi_axes must be floats")
        if any(sa <= 0 for sa in semi_axes):
            raise ValueError("All dimensions of semi_axes must be positive")
        if not hasattr(rot_q, "__len__") or (len(rot_q) != 4):
            raise TypeError(
                "rot_q must be a np.ndarray of four floats (w, x, y and z)"
            )
        if rot_q.dtype != float:
            raise TypeError("All dimensions of rot_q must be floats")

        self.__center = np.array([float(c) for c in center])
        self.__rot_q = np.array([float(q) for q in rot_q])
        self.__a, self.__b, self.__c = float(semi_axes[0]), float(semi_axes[1]), float(semi_axes[2])
        self._Mb__build()

    def _Mb__build(self):

        # Input parsing
        t_v, s_v = .5 * self._Mb__thick / self._Mb__v_size, self._Mb__layer_s / self._Mb__v_size
        a_v, b_v, c_v = self.__a / self._Mb__v_size, self.__b / self._Mb__v_size, self.__c / self._Mb__v_size
        ao_v, bo_v, co_v = a_v + t_v, b_v + t_v, c_v + t_v
        ai_v, bi_v, ci_v = a_v - t_v, b_v - t_v, c_v - t_v
        ao_v_p1, bo_v_p1, co_v_p1 = ao_v + 1, bo_v + 1, co_v + 1
        ao_v_m1, bo_v_m1, co_v_m1 = ao_v - 1, bo_v - 1, co_v - 1
        ai_v_p1, bi_v_p1, ci_v_p1 = ai_v + 1, bi_v + 1, ci_v + 1
        ai_v_m1, bi_v_m1, ci_v_m1 = ai_v - 1, bi_v - 1, ci_v - 1
        p0_v = self.__center / self._Mb__v_size

        # Generating the grid
        dx, dy, dz = float(self._Mb__voi_shape[0]), float(self._Mb__voi_shape[1]), float(self._Mb__voi_shape[2])
        dx2, dy2, dz2 = math.floor(.5 * dx), math.floor(.5 * dy), math.floor(.5 * dz)
        p0_v[0] -= dx2
        p0_v[1] -= dy2
        p0_v[2] -= dz2
        x_l, y_l, z_l = -dx2, -dy2, -dz2
        x_h, y_h, z_h = -dx2 + dx, -dy2 + dy, -dz2 + dz
        X, Y, Z = np.meshgrid(np.arange(x_l, x_h), np.arange(y_l, y_h), np.arange(z_l, z_h), indexing='xy')

        # Mask generation
        R_o = ((X - p0_v[0]) / ao_v) ** 2 + ((Y - p0_v[1]) / bo_v) ** 2 + ((Z - p0_v[2]) / co_v) ** 2
        R_i = ((X - p0_v[0]) / ai_v) ** 2 + ((Y - p0_v[1]) / bi_v) ** 2 + ((Z - p0_v[2]) / ci_v) ** 2
        self._Mb__mask = tomo_rotate(np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0)
        if self._Mb__mask.sum() == 0:
            raise MbError("Generated membrane has zero volume. Try changing its parameters.")

        # Surface generation
        R_i = ((X - p0_v[0]) / a_v) ** 2 + ((Y - p0_v[1]) / b_v) ** 2 + ((Z - p0_v[2]) / c_v) ** 2
        R_i = tomo_rotate(R_i, self.__rot_q, mode='reflect')
        self._Mb__surf = iso_surface(R_i, 1)
        add_sfield_to_poly(self._Mb__surf, self._Mb__mask, 'mb_mask', dtype='int', interp='NN', mode='points')
        # lio.save_vtp(self._Mb__surf, './out/hold.vtp')
        self._Mb__surf = poly_threshold(self._Mb__surf, 'mb_mask', mode='points', low_th=.5)
        # lio.save_vtp(self._Mb__surf, './out/hold2.vtp')

        # Outer layer
        R_o = ((X - p0_v[0]) / ao_v_p1) ** 2 + ((Y - p0_v[1]) / bo_v_p1) ** 2 + ((Z - p0_v[2]) / co_v_p1) ** 2
        R_i = ((X - p0_v[0]) / ao_v_m1) ** 2 + ((Y - p0_v[1]) / bo_v_m1) ** 2 + ((Z - p0_v[2]) / co_v_m1) ** 2
        G = tomo_rotate(np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0)

        # Inner layer
        R_o = ((X - p0_v[0]) / ai_v_p1) ** 2 + ((Y - p0_v[1]) / bi_v_p1) ** 2 + ((Z - p0_v[2]) / ci_v_p1) ** 2
        R_i = ((X - p0_v[0]) / ai_v_m1) ** 2 + ((Y - p0_v[1]) / bi_v_m1) ** 2 + ((Z - p0_v[2]) / ci_v_m1) ** 2
        G += tomo_rotate(np.logical_and(R_i >= 1, R_o <= 1), self.__rot_q, order=0)

        # Smoothing
        self._Mb__density = lin_map(density_norm(sp.ndimage.gaussian_filter(G.astype(float), s_v), inv=True), ub=0, lb=1)


    def __str__(self):
        center_str = f"({self.__center[0]:.2f}, {self.__center[1]:.2f}, {self.__center[2]:.2f})"
        semi_axes_str = f"({self.__a:.2f}, {self.__b:.2f}, {self.__c:.2f})"
        quat_str = f"({self.__rot_q[0]:.2f}, {self.__rot_q[1]:.2f}, {self.__rot_q[2]:.2f}, {self.__rot_q[3]:.2f})"
        return f"Membrane(Ellipsoid), thickness={self._Mb__thick:.3f} Å, layer sigma={self._Mb__layer_s:.3f} Å, center={center_str}, semi_axes={semi_axes_str}, rotation_quaternion={quat_str}"


@MbFactory.register("ellipsoid")
class EllipGen(MbGen):
    """
    Ellipsoidal membrane generator class
    """

    _ECC_MAX_TRIES = int(1e6)

    def __init__(
        self,
        thick_rg: tuple[float, float],
        layer_s_rg: tuple[float, float],
        occ_rg: tuple[float, float],
        over_tol: float,
        mb_den_cf_rg: tuple[float, float],
        max_ecc: float,
        min_axis: float,
        max_axis: float = None,
        
    ) -> None:
        """
        Constructor

        Args:
            thick_rg (tuple[float, float]): tuple with the min and max thickness values.
            layer_s_rg (tuple[float, float]): tuple with the min and max layer sigma values.
            occ_rg (tuple[float, float]): tuple with the min and max occupancy values.
            over_tol (float): overlap tolerance for the membrane set.
            mb_den_cf_rg (tuple[float, float]): tuple with the min and max membrane density contrast values.
            min_axis (float): minimum axis length of the ellipsoid in angstroms.
            max_axis (float, optional): maximum axis length of the ellipsoid in angstroms.

        Returns:
            None
        """

        super(EllipGen, self).__init__(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
        )
        self.__max_ecc = max_ecc
        self.__min_axis, self.__max_axis = min_axis, max_axis

    @classmethod
    def from_params(cls, params: dict) -> MbGen:
        """
        Creates a EllipGen object from a dictionary of parameters

        Args:
            params (dict): dictionary with the membrane parameters

        Returns:
            EllipGen: EllipGen object
        """

        thick_rg = params.get("MB_THICK_RG", (25.0, 35.0))
        layer_s_rg = params.get("MB_LAYER_S_RG", (0.5, 2.0))
        occ_rg = params.get("MB_OCC_RG", (0.001, 0.003))
        over_tol = params.get("MB_OVER_TOL", 0.0)
        mb_den_cf_rg = params.get("MB_DEN_CF_RG", (0.3, 0.5))
        min_axis = params.get("MB_MIN_AXIS", 75.0)
        max_axis = params.get("MB_MAX_AXIS", None)
        max_ecc = params.get("MB_MAX_ECC", 0.75)

        return cls(
            thick_rg=thick_rg,
            layer_s_rg=layer_s_rg,
            occ_rg=occ_rg,
            over_tol=over_tol,
            mb_den_cf_rg=mb_den_cf_rg,
            min_axis=min_axis,
            max_axis=max_axis,
            max_ecc=max_ecc,
        )

    def generate(self, voi_shape: tuple, v_size: float) -> MbEllipsoid:
        """
        Generates a ellipsoidal membrane with random parameters within the input volume of interest shape

        Args:
            voi_shape: shape of the volume of interest
            v_size: voxel size

        Returns:
            MbEllipsoid: generated ellipsoidal membrane
        """
        if self.__max_axis is None:
            self.__max_axis = math.sqrt(3) * max(voi_shape) * v_size

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

        for i in range(self._ECC_MAX_TRIES):

            axes = (
                gen_bounded_exp(
                    8.0 * self.__min_axis,
                    self.__min_axis,
                    self.__max_axis
                ),
                gen_bounded_exp(
                    8.0 * self.__min_axis,
                    self.__min_axis,
                    self.__max_axis
                ),
                gen_bounded_exp(
                    8.0 * self.__min_axis,
                    self.__min_axis,
                    self.__max_axis
                )
            )
            axes = np.sort(np.array(axes))[::-1]
            ecc1, ecc2 = (
                math.sqrt(1 - (axes[1]/axes[0])**2),
                math.sqrt(1 - (axes[2]/axes[0])**2),
            )
            if (ecc1 <= self.__max_ecc) and (ecc2 <= self.__max_ecc):
                axes = (axes[0], axes[1], axes[2])
                return MbEllipsoid(
                    voi_shape=voi_shape,
                    v_size=v_size,
                    thick=thick,
                    layer_s=layer_s,
                    center=center,
                    semi_axes=axes,
                    rot_q=gen_rand_unit_quaternion(),
                )
        raise MbError("Could not generate ellipsoid with the specified maximum eccentricity")

        
