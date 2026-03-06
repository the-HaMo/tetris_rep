from pathlib import Path
import random

import mrcfile
import numpy as np

from polnet.utils.affine import lin_map, poly_translate
from polnet.utils.arrays import vol_cube
import polnet.utils.poly as pp

class Pn:
    """
    Class representing a cytosolic protein entity
    """

    def __init__(
        self,
        entity_id: int,
        mmer_id: str,
        mmer_svol: np.ndarray,
        mmer_iso: float,
        pmer_l: float,
        pmer_l_max: float,
        pmer_occ: float
    ):
        self.__entity_id = entity_id
        self.__mmer_id = mmer_id
        self.__mmer_svol = mmer_svol
        self.__mmer_iso = mmer_iso
        self.__pmer_l = pmer_l
        self.__pmer_l_max = pmer_l_max
        self.__pmer_occ = pmer_occ
        self.__build()

    def __build(self):
        """Build the cytosolic protein entity structure.

        Raises:
            NotImplementedError: Cytosolic protein structure building not yet implemented.
        """
        raise NotImplementedError("Cytosolic protein structure building not yet implemented.")

class PnGen():
    """
    Class for generating cytosolic protein entities
    """ 

    def __init__(
        self,
        surf_dec: float,
        mmer_id: str,
        model: np.ndarray,
        mmer_iso: float,
        pmer_l: float,
        pmer_l_max: float,
        pmer_occ_rg: tuple[float, float],
        pmer_over_tol: float = 0.0
    ):
        self.__mmer_id = mmer_id
        self.__model = model
        self.__mmer_iso = mmer_iso
        self.__pmer_l = pmer_l
        self.__pmer_l_max = pmer_l_max
        self.__pmer_occ_rg = pmer_occ_rg
        self.__pmer_over_tol = pmer_over_tol
        self.__surf_dec = surf_dec
        self.__scale = 1.0
        self.__gen_model()

    @property
    def pmel_l(self) -> float:
        """Get the polymer length parameter.

        Returns:
            float: Polymer length parameter.
        """
        return self.__pmer_l
    
    @property
    def pmer_l_max(self) -> float:
        """Get the maximum polymer length parameter.

        Returns:
            float: Maximum polymer length parameter.
        """
        return self.__pmer_l_max

    def __gen_model(self):
        """Generate the cytosolic protein model.

        """
        self.__model = lin_map(self.__model, lb=0, ub=1)
        self.__model = vol_cube(self.__model)
        self.__model_mask = self.__model < self.__mmer_iso
        self.__model[self.__model_mask] = 0.0
        self.__model_surf = pp.iso_surface(
            self.__model,
            self.__mmer_iso,
            closed=False,
            normals=None
        )
        self.__model_surf = pp.poly_decimate(
            self.__model_surf,
            self.__surf_dec
        )
        self.__model_center = 0.5 * np.asarray(self.__model.shape, dtype=float)
        self.__model_surf  = poly_translate(
            self.__model_surf,
            -self.__model_center
        )

    def set_scale(self, scale: float):
        self.__model_surf = pp.poly_scale(self.__model_surf, scale/self.__scale)
        self.__scale = scale

    @property
    def surf_diam(self) -> float:
        """Get the surface diameter of the cytosolic protein model.

        Returns:
            float: Surface diameter of the cytosolic protein model.
        """
        return pp.poly_diam(self.__model_surf)

    def rnd_occ(self) -> float:
        """Generate a random occupancy value within the specified range.

        Returns:
            float: A random occupancy value.
        """
        return random.uniform(self.__pmer_occ_rg[0], self.__pmer_occ_rg[1])
    
    @property
    def pmer_l(self) -> float:
        """Get the polymer length parameter.

        Returns:
            float: Polymer length parameter.
        """
        return self.__pmer_l

    @property
    def over_tolerance(self) -> float:
        """Get the overlap tolerance parameter.

        Returns:
            float: Overlap tolerance parameter.
        """
        return self.__pmer_over_tol
    
    @property
    def surf(self):
        """Get the surface representation of the cytosolic protein model.

        Returns:
            pp.PolyData: Surface representation of the cytosolic protein model.
        """
        return self.__model_surf
    
    @property
    def model(self) -> np.ndarray:
        """Get the volumetric model of the cytosolic protein.

        Returns:
            np.ndarray: Volumetric model of the cytosolic protein.
        """
        return self.__model
    
    @property
    def mask(self) -> np.ndarray:
        """Get the mask of the cytosolic protein model.

        Returns:
            np.ndarray: Mask of the cytosolic protein model.
        """
        return self.__model_mask
    
    @property
    def svol(self) -> np.ndarray:
        """Get the volumetric model of the cytosolic protein.

        Returns:
            np.ndarray: Volumetric model of the cytosolic protein.
        """
        return self.__model < self.__mmer_iso
    
    @classmethod
    def from_params(cls, params: dict, data_path: Path, surf_dec: float) -> 'PnGen':
        """Create a PnGen instance from a parameters dictionary.

        Args:
            params (dict): Dictionary containing the parameters for cytosolic protein generation.
            data_path (Path): Path to the data directory containing the template files.

        Returns:
            PnGen: An instance of the PnGen class.
        """

        # Convert relative path to absolute path
        mmer_path = params["MMER_SVOL"]
        if mmer_path.startswith("/"):
            mmer_path = "." + mmer_path
        mmer_path = data_path / mmer_path

        # Check if PMER_OCC is a float or a tuple
        if isinstance(params["PMER_OCC"], float):
            params["PMER_OCC"] = (params["PMER_OCC"], params["PMER_OCC"])

        try:
            mrc = mrcfile.open(mmer_path, permissive=True, mode="r+")
            mmer_svol = np.swapaxes(mrc.data, 0, 2)
            mrc.close()
            return cls(
                surf_dec=surf_dec,
                mmer_id=params["MMER_ID"],
                model=mmer_svol,
                mmer_iso=params["MMER_ISO"],
                pmer_l=params["PMER_L"],
                pmer_l_max=params["PMER_L_MAX"],
                pmer_occ_rg=params["PMER_OCC"],
                pmer_over_tol=params.get("PMER_OVER_TOL", 0.0)
            )
        except Exception as e:
            raise FileNotFoundError(f"Error loading mmer_svol file {mmer_path}: {e}")

    def generate(voi_shape: tuple[int, int, int], v_size: float) -> Pn:
        """Generate a cytosolic protein entity.

        Args:
            voi_shape (tuple[int, int, int]): Shape of the volume of interest (VOI) in voxels.
            v_size (float): Voxel size in Angstroms.

        Returns:
            Pn: Generated cytosolic protein entity.
        """
        raise NotImplementedError("Cytosolic protein generation not yet implemented.")
    
class PnError(Exception):
    """Custom exception for cytosolic protein generation errors."""
    def __init__(self, message: str):
        super().__init__(message)