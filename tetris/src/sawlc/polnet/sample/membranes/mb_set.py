import sys

import numpy as np
import vtk

from .mb import Mb, MbError, MbGen
from polnet.utils.poly import poly_scale

class MbSet:
    """Class for modelling a set of membranes (same kind) within a tomogram."""

    def __init__(
        self,
        voi: np.ndarray,
        v_size: float,
        gen_rnd_surfs: MbGen,
        bg_voi: np.ndarray = None,
        max_mbtries: int = 10,
        grow: int = 0,
    ) -> None:
        """
        Constructor

        Setting up the parameters for generating a set of membranes within a tomogram.

        Args:
            voi (np.ndarray): A 3D numpy array to define a VOI (Volume Of Interest) for membranes.
            v_size (float): Voxel size (in angstroms).
            gen_rnd_surfs: An object that inherits from SurfGen class to generate random instances with
                             membrane surface parameters, therefore the objects class determine the shape of the membranes
                             generated.
            bg_voi (np.ndarray, optional): Background VOI (Default None). If present, membrane areas in this VOI will be removed and
                                            not considered for overlapping. It must be binary with the same shape of voi.
            max_mbtries (int, optional): Maximum number of tries to insert a membrane. Defaults to 10.
            grow (int, optional): Number of voxel to grow the VOI.
        """
        if voi is None or not isinstance(voi, np.ndarray) or voi.ndim != 3 or voi.dtype != bool:
            raise ValueError("voi must be a 3D numpy array of boolean type.")
        if bg_voi is not None:
            if not isinstance(bg_voi, np.ndarray) or bg_voi.shape != voi.shape or bg_voi.dtype != bool:
                raise ValueError("bg_voi must be a 3D numpy array of boolean type with the same shape as voi.")
        if not isinstance(v_size, (int, float)) or v_size <= 0:
            raise ValueError("v_size must be a positive number.")
        if not isinstance(gen_rnd_surfs, MbGen):
            raise ValueError("gen_rnd_surfs must be an instance of a class that inherits from MbGen.")
        if not isinstance(max_mbtries, int) or max_mbtries <= 0:
            raise ValueError("max_mbtries must be a positive integer.")
        if not isinstance(grow, int) or grow < 0:
            raise ValueError("grow must be a non-negative integer.")

        # Variables assignment
        self.__voi = voi
        self.__bg_voi = bg_voi
        self.__v_size = v_size
        self.__density = np.zeros(shape=voi.shape, dtype=np.float16)
        self.__mask = np.zeros(shape=voi.shape, dtype=bool)
        self.__surfs, self.__app_vtp = (
            vtk.vtkPolyData(),
            vtk.vtkAppendPolyData(),
        )
        self.__count_mbs = 0
        self.__gen_rnd_surfs = gen_rnd_surfs
        self.__max_mbtries = max_mbtries
        self.__grow = grow

    @property
    def vol(self) -> float:
        """Get the volume of the VOI."""
        return float(self.__voi.sum()) * self.__v_size * self.__v_size * self.__v_size

    @property
    def mb_occupancy(self) -> float:
        """Get the membrane occupancy within the VOI."""
        return self.__mask.sum() / np.prod(
            np.asarray(self.__voi.shape, dtype=float)
        )

    @property
    def voi(self) -> np.ndarray:
        """Get the VOI with the membranes inserted."""
        return self.__voi

    @property
    def density(self) -> np.ndarray:
        """Get the density values with the membranes within the VOI.

        Returns:
            np.ndarray: The density values.
        """
        return self.__density

    @property
    def mask(self) -> np.ndarray:
        """Get the mask of the membranes within the VOI.

        Returns:
            np.ndarray: The mask.
        """
        return self.__mask

    @property
    def vtp(self) -> vtk.vtkPolyData:
        """Get the set of membranes as a vtkPolyData with their surfaces.

        Returns:
            vtk.vtkPolyData: The set of membranes.
        """
        return self.__surfs

    @property
    def num_mbs(self) -> int:
        """Get the number of membranes in the set.

        Returns:
            int: The number of membranes.
        """
        return self.__count_mbs

    def check_overlap(self, mb: Mb, over_tolerance: float) -> bool:
        """
        Determines if the membrane overlaps with any member within the membranes set.

        Args:
            mb (Mb): Input Membrane to check for the overlapping.
            over_tolerance (float): Overlapping tolerance (percentage of membrane voxel overlapping).

        Returns:
            bool: True if there is an overlap, False otherwise.
        """

        # Checking the overlapping with the set ground truth and VOI
        mb_mask = mb.mask
        available = np.logical_and(self.__voi, ~self.__mask)
        if self.__bg_voi is not None:
            mb_mask = np.logical_and(mb_mask, self.__bg_voi)
        tomo_over = np.logical_and(
            mb_mask,
            ~available
        )

        return (tomo_over.sum() / mb.vol) > over_tolerance

    def insert_mb(
        self,
        mb: Mb,
        merge: str = "min",
        over_tolerance: float = None,
        check_vol: bool = True,
        grow: int = 0,
    ) -> None:
        """
        Insert the membrane into the set (density, mask and surf).

        Args:
            mb (Mb): Input membrane (Mb) object.
            merge (str, optional): Merging mode for density insertion, valid: 'min' (default), 'max', 'sum' and 'insert'.
            over_tolerance (float, optional): Overlapping tolerance (percentage of membrane voxel overlapping), if None then disabled.
            check_vol (bool, optional): If True (default), check for volume consistency before insertion.
            grow (int, optional): Number of voxels to grow the membrane before insertion.

        Raises:
            MbError: MbError if the membrane can not be inserted.

        Returns:
            None
        """
        if check_vol and (mb.vol <= 0):
            raise MbError("Membrane volume is zero or negative, cannot be inserted.")
        if (over_tolerance is None) or (not self.check_overlap(mb, over_tolerance)):
            # Density tomogram insertion
            mb.insert_density_svol(self.__density, merge=merge, mode='tomo')
            # Mask
            mb.insert_density_svol(self.__mask, merge='max', mode='mask')
            # VOI
            mb.insert_density_svol(self.__voi, merge='min', mode='voi', grow=grow)
            # Surfaces insertion
            self.__app_vtp.AddInputData(mb.vtp)
            self.__app_vtp.Update()
            self.__surfs = poly_scale(self.__app_vtp.GetOutput(), self.__v_size)
            self.__count_mbs += 1
        else:
            raise MbError("Membrane overlaps with the set, cannot be inserted.")

    def build_set(self, verbosity: bool = False) -> None:
        """
        Build a set of membranes and insert them in a tomogram and a vtkPolyData object.

        Args:
            verbosity (bool, optional): If True, the output message with info about the membranes generated is printed. Defaults to False.

        Returns:
            None
        """

        # Initialization
        count_mb = 1
        count_exp = 0
        cf = self.__gen_rnd_surfs.rnd_cf()
        max_occ = self.__gen_rnd_surfs.rnd_occ()
        over_tolerance = self.__gen_rnd_surfs.over_tolerance

        while self.mb_occupancy < max_occ:
            try:
                hold_mb = self.__gen_rnd_surfs.generate(voi_shape=self.__voi.shape, v_size=self.__v_size)

                # Background masking
                if self.__bg_voi is not None:
                    hold_mb.masking(self.__bg_voi)

                self.insert_mb(
                    hold_mb,
                    merge="max",
                    over_tolerance=over_tolerance,
                    check_vol=True,
                    grow=self.__grow,
                )
                if verbosity:
                    print(
                        f"Membrane {count_mb} inserted, total occupancy: {self.mb_occupancy:.3f}, volume: {hold_mb.vol} Å. Details:\n\t{hold_mb}"
                    )
                count_mb += 1
            except MbError:
                count_exp += 1
                if count_exp == self.__max_mbtries:
                    print(
                        f"WARNING: more than {self.__max_mbtries} tries failed to insert a membrane!",
                        file=sys.stderr,
                    )
                    break
            count_exp = 0
        
        # Apply contrast factor to the density
        self.__density *= cf

        return None
    

