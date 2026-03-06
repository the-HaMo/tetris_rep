"""Module for synthetic sample representation.

This module defines the SyntheticSample class, which models a synthetic Cryo-ET sample. 
"""

from pathlib import Path
import sys
import numpy as np


from .membranes import MbFactory, MbSet
from .pns import PnGen, PnSAWLCNet, NetTetris

from polnet.utils import poly as pp

class SyntheticSample():
    """A model for a synthetic Cryo-ET sample.
    
    Attributes:
        shape (tuple): The shape of the sample.
        v_size (float): The voxel size of the sample in angstroms.
        offset (tuple): The offset for the sample.
        voi (np.ndarray): The volume of interest.
        voi_voxels (int): The number of voxels in the volume of interest.
        bg_voi (np.ndarray): The background volume of interest.
        labels (np.ndarray): The sample labels.
        density (np.ndarray): The sample density.
        poly_vtp: The polygonal representation of the sample.
        skel_vtp: The skeletal representation of the sample.
        structure_counts (dict): Counts of different structures in the sample.
        voxel_counts (dict): Voxel counts of different structures in the sample.

    Methods:

    """
    def __init__(self, shape: tuple, v_size: int, offset=(4, 4, 4)):
        """Constructor. 
        
        Args:
            id (int): The sample ID.
            shape (tuple): The shape of the sample.
            v_size (int): The voxel size of the sample in angstroms.
            offset (tuple, optional): The offset for the sample. Defaults to (4, 4, 4).
        """
        self.__shape = shape
        self.__v_size = v_size
        self.__offset = offset
        self.__voi = None
        self.__voi_voxels = 0
        self.__bg_voi = None
        self.__labels = None
        self.__density = None
        self.__poly_vtp = None
        self.__skel_vtp = None
        self.__mbs_vtp = None
        self.__structure_counts = None
        self.__voxel_counts = None
        self.__entity_id_counter = 1
        self.__output_labels = None
        self.reset()

    def reset(self):
        """Reset the sample to its initial state. Call this method to clear all data.

        """
        self.__voi = np.zeros(shape=self.__shape, dtype=bool)
        self.__voi[
            self.__offset[0]: self.__shape[0] - self.__offset[0],
            self.__offset[1]: self.__shape[1] - self.__offset[1],
            self.__offset[2]: self.__shape[2] - self.__offset[2]
        ] = True
        self.__voi_voxels = self.__voi.sum()
        self.__bg_voi = self.__voi.copy()
        self.__labels = np.zeros(shape=self.__shape, dtype=np.uint8)
        self.__density = np.zeros(shape=self.__shape, dtype=np.float32)
        self.__poly_vtp = None
        self.__skel_vtp = None
        self.__mbs_vtp = None
        self.__structure_counts = {}
        self.__voxel_counts = {}
        self.__output_labels = {
            'membrane': 1,
            "actin": 2,
            "microtubule": 3,
            "cprotein": 4
        }
        self.__entity_id_counter = 1

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def density(self) -> np.ndarray:
        return self.__density.copy()
    
    @property
    def labels(self) -> np.ndarray:
        return self.__labels.copy()
    
    @property
    def poly_vtp(self) -> np.ndarray:
        return self.__poly_vtp
    
    @property
    def skel_vtp(self) -> np.ndarray:
        return self.__skel_vtp
    
    @property
    def v_size(self) -> float:
        return self.__v_size

    def structure_count(self, type: str) -> int:
        """Get the structure count for a given type.

        Args:
            type (str): The type of structure to retrieve ('membrane').

        Returns:
            int: The structure count for the specified type.
        """
        if type not in self.__structure_counts:
            print(f"Warning: {type} structure count not found.", file=sys.stderr)
        return self.__structure_counts.get(type, 0)

    def add_set_membranes(
            self, 
            params: dict, 
            max_mbtries: int = 10,
            grow: int = 0,
            verbosity: bool = True
        ) -> None:
        """Generate and add a set of membranes to the sample. Parameters for the membrane generator class should be provided via the params dict.

        Args:
            params (dict): Parameters for the membrane generator class. Should include 'type' key.
            max_ntries (int, optional): Maximum number of tries to add the membrane. Defaults to 10.
            grow (int, optional): Number of voxels to grow the membrane mask in the VOI. Defaults to 0.
            verbosity (bool, optional): Verbosity flag. Defaults to True.

        Raises:
            KeyError: if 'MB_TYPE' key is not in params.

        Returns:
            None
        """

        if "MB_TYPE" not in params:
            raise KeyError("params must include 'MB_TYPE' key specifying the membrane type.")
    
        mb_type = params["MB_TYPE"]
        mb_generator = MbFactory.create(mb_type, params)

        set_mbs = MbSet(
            voi=self.__voi,
            bg_voi=self.__bg_voi,
            v_size=self.__v_size,
            gen_rnd_surfs=mb_generator,
            max_mbtries=max_mbtries,
            grow=grow,
        )

        set_mbs.build_set(verbosity=verbosity)

        # Tomo update
        self.__voi = set_mbs.voi
        hold_den = set_mbs.density
        hold_mask = set_mbs.mask
        self.__density = np.maximum(self.__density, hold_den)
        self.__labels[hold_mask] = self.__entity_id_counter
        if 'membrane' not in self.__structure_counts:
            self.__structure_counts['membrane'] = 0
            self.__voxel_counts['membrane'] = 0
        self.__structure_counts['membrane'] += set_mbs.num_mbs
        self.__voxel_counts['membrane'] += (self.__labels == self.__entity_id_counter).sum()
        hold_vtp = set_mbs.vtp
        # Adding labels to polydata
        pp.add_label_to_poly(hold_vtp, self.__entity_id_counter, "Entity", mode="both")
        pp.add_label_to_poly(hold_vtp, self.__output_labels['membrane'], "Type", mode="both")
        if self.__poly_vtp is None:
            self.__poly_vtp = hold_vtp
            self.__skel_vtp = hold_vtp
            self.__mbs_vtp = hold_vtp
        else:
            self.__poly_vtp = pp.merge_polys(self.__poly_vtp, hold_vtp)
            self.__skel_vtp = pp.merge_polys(self.__skel_vtp, hold_vtp)
            self.__mbs_vtp = pp.merge_polys(self.__mbs_vtp, hold_vtp)
        self.__entity_id_counter += 1
        
        return None
    
    def add_helicoidal_network(
            self, 
            params: dict, 
            verbosity: bool = True
        ) -> None:
        # TODO: Implement method to add helicoidal networks
        pass

    def add_set_cproteins(
            self, 
            params: dict, 
            data_path: Path,
            surf_dec: float = 0.9,
            mmer_tries: int = 20,
            pmer_tries: int = 100,
            verbosity: bool = True
        ) -> None:
        """Generate and add a set of cytosolic proteins to the sample. Parameters for the protein generator class should be provided via the params dict.
        
        Args:
            params (dict): Parameters for the protein generator class. Should include 'type' key.
            data_path (Path): Path to the data directory containing the model files.
            surf_dec (float, optional): Surface decimation factor. Defaults to 0.9.
            verbosity (bool, optional): Verbosity flag. Defaults to True.
        """
        pn_generator = PnGen.from_params(params, data_path=data_path, surf_dec=surf_dec)
        pn_generator.set_scale(self.__v_size)

        # set_pns = PnSAWLCNet(
        #     voi=self.__voi,
        #     v_size=self.__v_size,
        #     gen_rnd_cproteins=pn_generator,
        #     tries_mmer=mmer_tries,
        #     tries_pmer=pmer_tries,
        #     verbosity=verbosity
        # )

        print("AAAAA", pn_generator.svol.shape)
        set_pns = PnSAWLCNet(
        # set_pns = NetTetris(
            voi=self.__voi,
            v_size=self.__v_size,
            l_length = pn_generator.pmer_l * pn_generator.surf_diam,
            m_surf = pn_generator.surf,
            max_p_length = pn_generator.pmer_l_max,
            occ = pn_generator.rnd_occ(),
            over_tolerance = pn_generator.over_tolerance,
            poly=None,
            svol = pn_generator.svol,
            tries_mmer=mmer_tries,
            tries_pmer=pmer_tries
        )
        
        set_pns.build_network()

        set_pns.insert_density_svol(
            m_svol = pn_generator.mask,
            tomo = self.__voi,
            v_size = self.__v_size,
            merge = "min"
        )

        set_pns.insert_density_svol(
            m_svol = pn_generator.model,
            tomo = self.__density,
            v_size = self.__v_size,
            merge = "max"
        )

        hold_lbls = np.zeros(shape=self.__shape, dtype=np.uint8)
        set_pns.insert_density_svol(
            m_svol = np.invert(pn_generator.mask),
            tomo = hold_lbls,
            v_size = self.__v_size,
            merge = "max"
        )
        self.__labels[hold_lbls > 0] = self.__entity_id_counter
        counts = set_pns.get_num_mmers()
        self.__structure_counts['cprotein'] = counts
        self.__voxel_counts['cprotein'] = (self.__labels == self.__entity_id_counter).sum()
        
        hold_vtp = set_pns.get_vtp()
        hold_skel_vtp = set_pns.get_skel()
        pp.add_label_to_poly(
            poly = hold_vtp,
            lbl = self.__entity_id_counter,
            p_name = "Entity",
            mode = "both"
        )
        pp.add_label_to_poly(
            poly = hold_skel_vtp,
            lbl = self.__entity_id_counter,
            p_name = "Entity",
            mode = "both"
        )
        pp.add_label_to_poly(
            poly = hold_vtp,
            lbl = self.__output_labels['cprotein'],
            p_name = "Type",
            mode = "both"
        )
        pp.add_label_to_poly(
            poly = hold_skel_vtp,
            lbl = self.__output_labels['cprotein'],
            p_name = "Type",
            mode = "both"
        )
        if self.__poly_vtp is None:
            self.__poly_vtp = hold_vtp
            self.__skel_vtp = hold_skel_vtp
        else:
            self.__poly_vtp = pp.merge_polys(self.__poly_vtp, hold_vtp)
            self.__skel_vtp = pp.merge_polys(self.__skel_vtp, hold_skel_vtp)
        self.__entity_id_counter += 1

    def print_summary(self) -> None:
        """Prints a summary of the sample contents.

        Returns:
            None
        """
        print("Synthetic Sample Summary:")
        print(f"  Shape: {self.__shape}")
        print(f"  Voxel Size: {self.__v_size} Å")
        print(f"  VOI Voxels: {self.__voi_voxels}")
        print("  Structure Counts:")
        for struct_type, count in self.__structure_counts.items():
            print(f"    {struct_type}:")
            print(f"      Count: {count}")
            print(f"      Occupancy: {100.0* self.__voxel_counts.get(struct_type, 0) / self.__voi_voxels:.4f} %")

        
    
    

