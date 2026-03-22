import random

import numpy as np
import vtk

from polnet.sample.polymers import Network, SAWLC
from polnet.sample.polymers.swalc import SAWLCPoly
from polnet.utils.poly import *
from polnet.utils.affine import points_distance
from polnet.utils.spatial import *

class NetSAWLC(Network):
    """
    Class for generating a network of SAWLC polymers
    """

    def __init__(
        self,
        voi,
        v_size,
        l_length,
        m_surf,
        max_p_length,
        occ,
        over_tolerance=0,
        poly=None,
        svol=None,
        tries_mmer=50,
        tries_pmer=10,
        rots=None,
        rot_id=0,
        verbosity=False,
    ):
        """
        Construction

        :param voi: a 3D numpy array to define a VOI (Volume Of Interest) for polymers
        :param v_size: voxel size (default 1)
        :param l_length: polymer link length
        :param m_surf: monomer surf
        :param max_p_length: maximum polymer length
        :param gen_pol_lengths: a instance of a random generation model (random.PGen) to determine the achievable
        lengths for polymers
        :param occ: occupancy threshold in percentage [0, 100]%
        :param over_tolerance: fraction of overlapping tolerance for self avoiding (default 0, in range [0,1))
        :param poly: it allows to restrict monomer localizations to a polydata (e.g. a membrane)
        :param svol: monomer subvolume as a numpy ndarray (default None)
        :param off_svol: offset coordinates in voxels for shifting sub-volume monomer center coordinates (default None)
        :param tries_mmer: number of tries to place a monomer before starting a new polymer
        :param tries_pmer: number of tries to place a polymer
        :param rots: allow to externally control the rotations of the macromolecules, if not None (default), the
                     rotations (quaternions) are taken sequentially from the input array of rotations.
        :param rot_id: starting index for rots rotations array.
        """

        # Initialize abstract varibles
        super(NetSAWLC, self).__init__(voi, v_size, svol=svol)

        # Input parsing
        assert l_length > 0
        assert isinstance(m_surf, vtk.vtkPolyData)
        assert max_p_length >= 0
        assert (occ >= 0) and (occ <= 100)
        assert (over_tolerance >= 0) and (over_tolerance <= 100)
        if poly is not None:
            assert isinstance(poly, vtk.vtkPolyData)
        assert tries_mmer >= 1
        assert tries_pmer >= 1
        self.__rots, self.__rot_id = None, 0
        if rots is not None:
            assert len(rots) > 0 and len(rots[0]) == 4
            self.__rots = rots
            self.__rot_id = int(rot_id)

        # Variables assignment
        self.__l_length, self.__m_surf = l_length, m_surf
        self.__max_p_length = max_p_length
        self.__occ, self.__over_tolerance = occ*100, over_tolerance
        self.__poly = poly
        self.__tries_mmer = int(tries_mmer)
        self.__tries_pmer = int(tries_pmer)
        self.__verbosity = bool(verbosity)
        self.__poly_area = None
        if self.__poly is not None:
            self._Network__poly_area = poly_surface_area(self.__poly)

        # print("VOI shape =", self._Network__voi.shape)
        # print("voxel size =", self._Network__v_size)
        # print("# points =", self.__m_surf.GetNumberOfPoints())
        # print("l_length =", self.__l_length)
        # print("max_p_length =", self.__max_p_length)
        # print("target occupancy =", self.__occ)
        # print("monomer overlap tolerance =", self.__over_tolerance)
        # print("tries per monomer =", self.__tries_mmer)
        # print("tries per polymer =", self.__tries_pmer)

        # print("Available volume =", self._Network__voi.sum() * (self._Network__v_size ** 3), "nm^3")

    def build_network(self):
        """
        Add polymers following SAWLC model until an occupancy limit is passed

        :return:
        """

        c_try = 0
        iter_id = 0
        self._Network__pmer_fails = 0
        total_proteins = 0
        if self.__rots is not None:
            rot_id = self.__rot_id

        # Network loop
        while (c_try <= self.__tries_pmer) and (
            self._Network__pl_occ < self.__occ
        ):

            # Polymer initialization
            c_try += 1
            if self.__poly:
                p0 = np.asarray(
                    self.__poly.GetPoint(
                        random.randint(0, self.__poly.GetNumberOfPoints())
                    )
                )
            else:
                p0 = np.asarray(
                    (
                        self._Network__voi.shape[0]
                        * self._Network__v_size
                        * random.random(),
                        self._Network__voi.shape[1]
                        * self._Network__v_size
                        * random.random(),
                        self._Network__voi.shape[2]
                        * self._Network__v_size
                        * random.random(),
                    )
                )
            max_length = random.uniform(
                0, self.__max_p_length
            )
            hold_rot = None
            if self.__rots is not None:
                hold_rot = self.__rots[rot_id]
            if self.__poly is None:
                hold_polymer = SAWLC(
                    self.__l_length, self.__m_surf, p0, rot=hold_rot
                )
            else:
                hold_polymer = SAWLCPoly(
                    self.__poly,
                    self.__l_length,
                    self.__m_surf,
                    p0,
                    rot=hold_rot,
                )
            if hold_polymer.get_monomer(-1).overlap_voi(
                self._Network__voi,
                self._Network__v_size,
                over_tolerance=self.__over_tolerance,
            ):
                self._Network__pmer_fails += 1
                continue
            self.add_monomer_to_voi(
                hold_polymer.get_monomer(-1), self._Network__svol
            )
            if self.__rots is not None:
                if rot_id >= len(self.__rots) - 1:
                    rot_id = 0
                else:
                    rot_id += 1

            # Polymer loop
            cont_pol = 1
            not_finished = True
            while (hold_polymer.get_total_len() < max_length) and not_finished:
                hold_rot = None
                if self.__rots is not None:
                    hold_rot = self.__rots[rot_id]
                monomer_data = hold_polymer.gen_new_monomer(
                    self.__over_tolerance,
                    self._Network__voi,
                    self._Network__v_size,
                    fix_dst=random.uniform(
                        self.__l_length, 2 * self.__l_length
                    ),
                    rot=hold_rot,
                )

                cont_pol += 1

                if monomer_data is None:
                    if cont_pol >= self.__tries_mmer:
                        not_finished = False
                    else:
                        c_try += 1
                else:
                    new_len = points_distance(
                        monomer_data[0], hold_polymer.get_tail_point()
                    )
                    if hold_polymer.get_total_len() + new_len < max_length:
                        # ) and (monomer_data[3].overlap_voi(self._Network__voi, self._Network__v_size)):
                        hold_polymer.add_monomer(
                            monomer_data[0],
                            monomer_data[1],
                            monomer_data[2],
                            monomer_data[3],
                        )
                        self.add_monomer_to_voi(
                            hold_polymer.get_monomer(-1), self._Network__svol
                        )
                        hold_occ = self._Network__pl_occ + 100.0 * (
                            hold_polymer.get_vol() / self._Network__vol
                        )
                        if self.__rots is not None:
                            if rot_id >= len(self.__rots) - 1:
                                rot_id = 0
                            else:
                                rot_id += 1
                        if hold_occ >= self.__occ:
                            not_finished = False
                    else:
                        not_finished = False

            # Updating polymer
            if self.__poly is None:
                self.add_polymer(hold_polymer, occ_mode="volume")
                c_try = 0
            else:
                self.add_polymer(hold_polymer, occ_mode="area")
                c_try = 0

            iter_id += 1
            if self.__verbosity:
                print(
                    f"Paso {iter_id}: {hold_polymer.get_num_mmers()} proteinas insertadas. Ocupancia {self._Network__pl_occ:.4f}%"
                )
            total_proteins += hold_polymer.get_num_mmers()
            # print('build_network: new polymer added with ' + str(hold_polymer.get_num_monomers()) +
            #       ', length ' + str(hold_polymer.get_total_len()) + ' and occupancy ' +
            #       str(self._Network__pl_occ) + '%')

        print(f"Total proteinas insertadas: {total_proteins}")
        print(f"Pmer fails: {self._Network__pmer_fails}")
        # print('Exit with c_try=' + str(c_try) + ' and c_fails=' + str(self._Network__pmer_fails))


class NetTetris(Network):
    """
    Class for generating a network of polymers using Tetris grid-based placement.
    
    This algorithm places proteins on a regular grid instead of using stochastic
    Self-Avoiding Walk Linear Chain (SAWLC). 
    
    Phase 1: Basic grid-based placement
    Phase 2: Add noise_level for heterogeneity
    """

    def __init__(
        self,
        voi,
        v_size,
        l_length,
        m_surf,
        max_p_length,
        occ,
        over_tolerance=0,
        poly=None,
        svol=None,
        tries_mmer=50,
        tries_pmer=10,
        rots=None,
        rot_id=0,
    ):
        """
        Construction - same parameters as NetSAWLC for compatibility.
        
        :param voi: 3D numpy array for Volume Of Interest
        :param v_size: voxel size
        :param l_length: polymer link length
        :param m_surf: monomer surface
        :param max_p_length: maximum polymer length
        :param occ: occupancy threshold [0, 100]%
        :param over_tolerance: overlap tolerance for self-avoiding
        :param poly: restrict monomers to polydata (e.g. membrane)
        :param svol: monomer subvolume
        :param tries_mmer: tries per monomer
        :param tries_pmer: tries per polymer
        :param rots: external rotation control
        :param rot_id: starting rotation index
        """
        
        # Initialize abstract variables
        super(NetTetris, self).__init__(voi, v_size, svol=svol)

        # Input parsing (same as NetSAWLC)
        assert l_length > 0
        assert isinstance(m_surf, vtk.vtkPolyData)
        assert max_p_length >= 0
        assert (occ >= 0) and (occ <= 100)
        assert (over_tolerance >= 0) and (over_tolerance <= 100)
        if poly is not None:
            assert isinstance(poly, vtk.vtkPolyData)
        assert tries_mmer >= 1
        assert tries_pmer >= 1
        
        self.__rots, self.__rot_id = None, 0
        if rots is not None:
            assert len(rots) > 0 and len(rots[0]) == 4
            self.__rots = rots
            self.__rot_id = int(rot_id)

        # Variables assignment
        self.__l_length, self.__m_surf = l_length, m_surf
        self.__max_p_length = max_p_length
        self.__occ, self.__over_tolerance = occ, over_tolerance
        self.__poly = poly
        self.__tries_mmer = int(tries_mmer)
        self.__tries_pmer = int(tries_pmer)
        self.__poly_area = None
        if self.__poly is not None:
            self._Network__poly_area = poly_surface_area(self.__poly)

    def build_network(self, noise_level=0.7):
        """
        Add polymers using Tetris grid-based placement algorithm.
        
        This method places protein monomers on a regular grid for efficiency,
        then optionally adds noise for heterogeneity.
        
        Args:
            noise_level (float): Heterogeneity level (0.0-1.0, default 0.7 for Phase 1)
        
        Returns:
            dict: Statistics with keys:
                - 'proteins_placed': Number of proteins successfully placed
                - 'grid_positions': Total grid positions generated
                - 'collisions': Number of rejected positions due to VOI overlap
                - 'occupancy': Final occupancy percentage
        """
        
        # Estimate protein size from monomer surface
        bounds = self.__m_surf.GetBounds() # return [xMin, xMax, yMin, yMax, zMin, zMax]
        protein_diameter = max(
            bounds[1] - bounds[0], # X size
            bounds[3] - bounds[2], # Y size
            bounds[5] - bounds[4] # Z size
        )
        
        # Convert to voxel units
        protein_size_voxels = protein_diameter / self._Network__v_size
        
        # VOI shape in voxels
        voi_shape = self._Network__voi.shape
        
        # Generate grid positions (in voxels) with 1.2x spacing (20% safety margin) probally not needed
        grid_positions = generate_grid_positions(
            voi_shape=voi_shape,
            protein_size=protein_size_voxels,
            spacing_multiplier=1.2
        )
        
        statistics = {
            'proteins_placed': 0,
            'grid_positions': len(grid_positions),
            'collisions': 0,
            'occupancy': 0.0
        }
        
        # Track rotation ID if rotations provided
        if self.__rots is not None:
            rot_id = self.__rot_id
        
        # Place proteins on grid positions (working in voxel units)
        for grid_pos in grid_positions:
            # Add noise for heterogeneity (in voxel units)
            if noise_level > 0:
                pos_voxels = add_random_offset(
                    grid_pos,
                    offset_magnitude=noise_level,
                    protein_size=protein_size_voxels
                )
            else:
                pos_voxels = grid_pos
            
            # Check if position is valid (within VOI bounds)
            if not is_position_in_voi(
                pos_voxels,
                voi_shape,
                protein_size_voxels / 2
            ):
                statistics['collisions'] += 1
                continue
            
            # Try to create polymer starting at this position
            try:
                # Convert to physical units (nm) ONLY when creating polymer
                p0 = np.asarray([
                    pos_voxels[0] * self._Network__v_size,
                    pos_voxels[1] * self._Network__v_size,
                    pos_voxels[2] * self._Network__v_size
                ])
                
                hold_rot = None
                if self.__rots is not None:
                    hold_rot = self.__rots[rot_id]
                
                # Create single monomer polymer
                if self.__poly is None:
                    hold_polymer = SAWLC(
                        self.__l_length,
                        self.__m_surf,
                        p0,
                        rot=hold_rot
                    )
                else:
                    hold_polymer = SAWLCPoly(
                        self.__poly,
                        self.__l_length,
                        self.__m_surf,
                        p0,
                        rot=hold_rot,
                    )
                
                # Check if monomer overlaps with VOI
                if hold_polymer.get_monomer(-1).overlap_voi(
                    self._Network__voi,
                    self._Network__v_size,
                    over_tolerance=self.__over_tolerance,
                ):
                    statistics['collisions'] += 1
                    continue
                
                # Add monomer to VOI
                self.add_monomer_to_voi(
                    hold_polymer.get_monomer(-1),
                    self._Network__svol
                )
                
                # Add polymer to network
                if self.__poly is None:
                    self.add_polymer(hold_polymer, occ_mode="volume")
                else:
                    self.add_polymer(hold_polymer, occ_mode="area")
                
                statistics['proteins_placed'] += 1
                
                # Update rotation counter
                if self.__rots is not None:
                    rot_id = (rot_id + 1) % len(self.__rots)
                
                # Check if occupancy target reached
                if self._Network__pl_occ >= self.__occ:
                    break
                    
            except Exception:
                # Skip positions that fail polymer generation
                statistics['collisions'] += 1
                continue
        
        statistics['occupancy'] = self._Network__pl_occ
        return statistics
