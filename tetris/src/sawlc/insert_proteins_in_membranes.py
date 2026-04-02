"""
Insert proteins into existing membrane tomograms using Tetris-like algorithm
"""

import os
from pathlib import Path
import numpy as np
import time
import scipy.ndimage as ndi
import vtk

from polnet.tomogram import SynthTomo
from polnet.utils import lio
from polnet.utils import poly as pp

# Configuration
ROOT_PATH = Path(__file__).resolve().parents[2] / "data"
MEMBRANES_PATH = ROOT_PATH / "templates" / "membranes"

# Membrane files to process
MEMBRANE_FILES = [
    "tomo_mem_lbls_0.mrc",
    # "tomo_mem_lbls_1.mrc",
    # "tomo_mem_lbls_2.mrc",
    # "tomo_mem_lbls_3.mrc",
 ]

# Proteins to insert
PROTEINS_LIST = [
    "in_10A/4v4r_10A.pns",
    # "in_10A/3j9i_10A.pns",
    # "in_10A/5mrc_10A.pns",
    # "in_10A/4v7r_10A.pns",
    # "in_10A/2uv8_10A.pns",
    # "in_10A/4v94_10A.pns",
    # "in_10A/4cr2_10A.pns",
    # "in_10A/3qm1_10A.pns",
    # "in_10A/3h84_10A.pns",
    # "in_10A/3gl1_10A.pns",
    # "in_10A/3d2f_10A.pns",
    # "in_10A/3cf3_10A.pns",
    # "in_10A/2cg9_10A.pns",
    # "in_10A/1u6g_10A.pns",
    # "in_10A/1s3x_10A.pns",
    # "in_10A/1qvr_10A.pns",
]

# Output directory
OUT_DIR = ROOT_PATH / "data_generated" / "output" / "output_proteins_in_membranes"
os.makedirs(OUT_DIR, exist_ok=True)

VOI_VSIZE = 10  # A/vx
MEMBRANE_INTENSITY_SCALE = 0.35
PROTEIN_CLEARANCE_VOXELS = 6
CLEAN_INTERMEDIATE_FILES = True
EXPORT_VTP = True
PROTEIN_ISO_THRESHOLD_RATIO = 0.08
DEBUG_OCCUPANCY = True

# Set random seed for reproducibility
np.random.seed(42)


def add_uniform_poly_labels(poly, entity_id, type_id):
    n_cells = poly.GetNumberOfCells()
    n_points = poly.GetNumberOfPoints()

    arr_entity_cells = vtk.vtkIntArray()
    arr_entity_cells.SetName("Entity")
    arr_entity_cells.SetNumberOfValues(n_cells)
    for i in range(n_cells):
        arr_entity_cells.SetValue(i, int(entity_id))
    poly.GetCellData().AddArray(arr_entity_cells)

    arr_type_cells = vtk.vtkIntArray()
    arr_type_cells.SetName("Type")
    arr_type_cells.SetNumberOfValues(n_cells)
    for i in range(n_cells):
        arr_type_cells.SetValue(i, int(type_id))
    poly.GetCellData().AddArray(arr_type_cells)

    arr_entity_points = vtk.vtkIntArray()
    arr_entity_points.SetName("Entity")
    arr_entity_points.SetNumberOfValues(n_points)
    for i in range(n_points):
        arr_entity_points.SetValue(i, int(entity_id))
    poly.GetPointData().AddArray(arr_entity_points)

    arr_type_points = vtk.vtkIntArray()
    arr_type_points.SetName("Type")
    arr_type_points.SetNumberOfValues(n_points)
    for i in range(n_points):
        arr_type_points.SetValue(i, int(type_id))
    poly.GetPointData().AddArray(arr_type_points)


def insert_proteins_in_membrane(membrane_mrc_path, proteins_list, output_dir, membrane_id):
    """
    Load membrane from MRC file and insert proteins into it
    
    Args:
        membrane_mrc_path: Path to membrane MRC file
        proteins_list: List of protein file paths
        output_dir: Output directory for results
        membrane_id: ID of the membrane for naming
    """
    print(f"\n{'='*60}")
    print(f"Processing membrane: {os.path.basename(membrane_mrc_path)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load membrane from MRC
    print(f"Loading membrane from: {membrane_mrc_path}")
    try:
        membrane_volume = lio.load_mrc(str(membrane_mrc_path))
        print(f"Membrane shape: {membrane_volume.shape}")
        print(f"Membrane value range: [{membrane_volume.min()}, {membrane_volume.max()}]")
    except Exception as e:
        print(f"ERROR loading membrane: {e}")
        return None
    
    # Create SynthTomo with loaded membrane as base
    # Note: We're using the membrane as a starting point and adding proteins to it
    try:
        tomo = SynthTomo(
            id=membrane_id,
            mbs_file_list=[],  # No need to generate membranes
            hns_file_list=[],
            pns_file_list=proteins_list,
            pms_file_list=[],
        )
        
        # Generate proteins in the membrane volume
        tomo.gen_sample(
            data_path=ROOT_PATH,
            shape=membrane_volume.shape,
            v_size=VOI_VSIZE,
            offset=(0, 0, 0),
            verbosity=True
        )
        
        # Save SAWLC outputs first (density, labels, VTP) using the public API
        output_dir_membrane = Path(output_dir) / f"membrane_{membrane_id:02d}"
        os.makedirs(output_dir_membrane, exist_ok=True)
        tomo.save_tomo(output_folder=output_dir_membrane)

        # Read generated density and combine it with the provided membrane volume
        generated_den_path = output_dir_membrane / f"tomo_{membrane_id:03d}_den.mrc"
        generated_density = lio.load_mrc(str(generated_den_path)).astype(np.float32)
        generated_lbl_path = output_dir_membrane / f"tomo_{membrane_id:03d}_lbl.mrc"
        generated_labels = lio.load_mrc(str(generated_lbl_path)).astype(np.int32)

        print("\nCombining membrane with proteins...")
        membrane_mask = membrane_volume > 0
        # Distance map from membrane voxels; proteins are only kept farther than clearance.
        dist_to_membrane = ndi.distance_transform_edt(~membrane_mask)
        protein_allowed_mask = dist_to_membrane > PROTEIN_CLEARANCE_VOXELS

        max_protein = float(generated_density.max()) if generated_density.size > 0 else 0.0
        membrane_component = membrane_mask.astype(np.float32) * (MEMBRANE_INTENSITY_SCALE * max(max_protein, 1.0))
        proteins_density = generated_density.copy()
        proteins_density[~protein_allowed_mask] = 0.0
        combined_volume = proteins_density + membrane_component
        
        print(f"Combined volume range: [{combined_volume.min()}, {combined_volume.max()}]")
        
        # Save combined density
        output_file = output_dir_membrane / f"tomo_mem{membrane_id:02d}_with_proteins_den.mrc"
        lio.write_mrc(
            combined_volume,
            str(output_file),
            v_size=VOI_VSIZE,
            dtype=np.float32
        )
        print(f"Saved combined density to: {output_file}")

        labels = np.zeros(membrane_mask.shape, dtype=np.uint8)
        labels[membrane_mask] = 1
        labels[(proteins_density > 0) & (labels == 0)] = 2
        labels_file = output_dir_membrane / f"tomo_mem{membrane_id:02d}_labels.mrc"
        lio.write_mrc(labels, str(labels_file), v_size=VOI_VSIZE, dtype=np.uint8)
        print(f"Saved labels to: {labels_file}")

        if DEBUG_OCCUPANCY:
            total_voxels = membrane_mask.size
            membrane_occ = np.count_nonzero(membrane_mask) / total_voxels
            print(f"[DEBUG] Membrane occupancy before proteins: {membrane_occ*100:.4f}%")

            # In this script SynthTomo is created with mbs_file_list=[], so protein entity ids start at 1.
            for p_idx, protein_path in enumerate(proteins_list, start=1):
                entity_id = p_idx
                protein_name = os.path.basename(protein_path)
                # Apply the same membrane-exclusion mask used for final outputs.
                protein_vox = np.logical_and(generated_labels == entity_id, protein_allowed_mask)
                proteins_accum_vox = np.logical_and(
                    np.logical_and(generated_labels >= 1, generated_labels <= entity_id),
                    protein_allowed_mask,
                )

                inserted_for_type = np.count_nonzero(protein_vox)
                protein_occ = np.count_nonzero(proteins_accum_vox) / total_voxels
                total_occ = (np.count_nonzero(membrane_mask) + np.count_nonzero(proteins_accum_vox)) / total_voxels
                print(
                    f"[DEBUG] After type {p_idx:02d} ({protein_name}): "
                    f"inserted_vox={inserted_for_type}, "
                    f"proteins_occ={protein_occ*100:.4f}%, "
                    f"total_occ={total_occ*100:.4f}%"
                )

        final_vtp_file = output_dir_membrane / f"tomo_mem{membrane_id:02d}_final.vtp"
        if EXPORT_VTP:
            membrane_poly = pp.iso_surface(membrane_mask.astype(np.float32), th=0.5)
            add_uniform_poly_labels(membrane_poly, entity_id=1, type_id=1)

            proteins_threshold = max(float(proteins_density.max()) * PROTEIN_ISO_THRESHOLD_RATIO, 1e-6)
            proteins_mask = proteins_density > proteins_threshold
            # Safety pass: ensure no protein voxel remains in the exclusion band.
            proteins_mask = np.logical_and(proteins_mask, protein_allowed_mask)
            proteins_mask = ndi.binary_opening(proteins_mask, iterations=1)

            if np.any(proteins_mask):
                proteins_poly = pp.iso_surface(proteins_mask.astype(np.float32), th=0.5)
                add_uniform_poly_labels(proteins_poly, entity_id=2, type_id=4)
                final_poly = pp.merge_polys(proteins_poly, membrane_poly)
            else:
                final_poly = membrane_poly

            lio.save_vtp(final_poly, str(final_vtp_file))
            print(f"Saved final VTP to: {final_vtp_file}")

        if CLEAN_INTERMEDIATE_FILES:
            keep_names = {
                output_file.name,
                labels_file.name,
            }
            if EXPORT_VTP:
                keep_names.add(final_vtp_file.name)
            for file_path in output_dir_membrane.iterdir():
                if file_path.is_file() and file_path.name not in keep_names:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass
            print(f"Kept only: {sorted(keep_names)}")

        print(f"Saved outputs in: {output_dir_membrane}")
        
        elapsed = time.time() - start_time
        print(f"Processing time: {elapsed:.2f}s")
        
        return {
            'membrane_id': membrane_id,
            'output_dir': output_dir_membrane
        }
        
    except Exception as e:
        print(f"ERROR processing membrane: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution"""
    
    print("\n" + "="*60)
    print("PROTEIN INSERTION INTO EXISTING MEMBRANES")
    print("="*60)
    print(f"Membrane directory: {MEMBRANES_PATH}")
    print(f"Number of membranes: {len(MEMBRANE_FILES)}")
    print(f"Number of protein types: {len(PROTEINS_LIST)}")
    print(f"Output directory: {OUT_DIR}")
    
    total_start = time.time()
    
    for idx, mem_file in enumerate(MEMBRANE_FILES):
        membrane_path = MEMBRANES_PATH / mem_file
        
        if not membrane_path.exists():
            print(f"\nWARNING: Membrane file not found: {membrane_path}")
            continue
        
        result = insert_proteins_in_membrane(
            membrane_path,
            PROTEINS_LIST,
            OUT_DIR,
            membrane_id=idx
        )
        
        if result:
            print(f"✓ Successfully processed membrane {idx}")
        else:
            print(f"✗ Failed to process membrane {idx}")
    
    total_time = time.time() - total_start
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    
    print("\n" + "="*60)
    print(f"TOTAL PROCESSING TIME: {total_minutes}min {total_seconds:.2f}s")
    print(f"Output directory: {OUT_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
