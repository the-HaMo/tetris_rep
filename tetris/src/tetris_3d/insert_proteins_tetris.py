import os, sys, time, numpy as np
from pathlib import Path
from tetris import Tetris3D
from image_processing_3d import ImageProcessing3D
from parser_3d import Parser3D
import lio

# CONFIGURACIÓN
ROOT_PATH = Path(__file__).resolve().parents[2] / "data"
MEMBRANES_PATH = ROOT_PATH / "templates" / "membranes"

MEMBRANE_FILES = [
    "tomo_mem_lbls_0.mrc",
    # "tomo_mem_lbls_1.mrc",
    # "tomo_mem_lbls_2.mrc",
    # "tomo_mem_lbls_3.mrc",
]

PROTEINS_LIST = [
    "in_10A/4v4r_10A.pns",
    # "in_10A/3j9i_10A.pns",
    "in_10A/5mrc_10A.pns",
    # "in_10A/4v7r_10A.pns",
    # "in_10A/2uv8_10A.pns",
    "in_10A/4v94_10A.pns",
    "in_10A/4cr2_10A.pns",
    # "in_10A/3qm1_10A.pns",
    # "in_10A/3h84_10A.pns",
    # "in_10A/3gl1_10A.pns",
    "in_10A/3d2f_10A.pns",
    # "in_10A/3cf3_10A.pns",
    # "in_10A/2cg9_10A.pns",
    # "in_10A/1u6g_10A.pns",
    # "in_10A/1s3x_10A.pns",
    # "in_10A/1qvr_10A.pns",
]


OUT_DIR = ROOT_PATH / "data_generated" / "output" / "output_proteins_tetris_final"
PROTEIN_ISO_THRESHOLD_RATIO = 0.08
MEMBRANE_INTENSITY_SCALE = 0.35
VOI_VSIZE = 10

def _crop_volume(vol, threshold):
    """Elimina el aire del archivo original para permitir que la densidad llegue al borde."""
    coords = np.argwhere(vol > threshold)
    if coords.size == 0: return vol
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return vol[z0:z1, y0:y1, x0:x1]

def _pick_seed(allowed_mask, output_volume, threshold, box_size):
    """Muestreo uniforme para llenar bordes igual que el centro."""
    half = box_size // 2
    z_dim, y_dim, x_dim = output_volume.shape
    empty = allowed_mask & (output_volume <= threshold)
    viable = np.zeros_like(empty, dtype=bool)
    # Permite que el centro esté en el radio exacto (el borde de la caja toca 0)
    viable[half:z_dim-half, half:y_dim-half, half:x_dim-half] = empty[half:z_dim-half, half:y_dim-half, half:x_dim-half]
    candidates = np.argwhere(viable)
    if len(candidates) == 0: return None
    return tuple(candidates[np.random.randint(0, len(candidates))])

def insert_proteins_in_membrane(membrane_mrc_path, proteins_list, output_dir, membrane_id):
    print(f"\nProcessing Tomogram: {os.path.basename(membrane_mrc_path)}")
    start_time = time.time()
    membrane_volume = lio.load_mrc(str(membrane_mrc_path)).astype(np.float32)
    allowed_mask = ~(membrane_volume > 0)
    
    molecules = []
    for p_path in proteins_list:
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p_path), str(ROOT_PATH))
        # EL SECRETO: Recortar la proteína para quitar el padding interno
        vol_cropped = _crop_volume(vol, vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        molecules.append((os.path.basename(p_path), vol_cropped))

    global_threshold = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris_obj = Tetris3D(dimensions=membrane_volume.shape, threshold=global_threshold)
    tetris_obj.output_volume[~allowed_mask] = 500.0 

    total_inserted = 0
    for name, volume in molecules:
        box_size = max(volume.shape)
        current_target = _pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)
        consecutive_failures = 0
        
        while consecutive_failures < 10:
            if current_target is None: break
            rotated, _ = ImageProcessing3D.randomly_rotate(volume)
            rotated_bin = ImageProcessing3D.smooth_and_binarize(rotated, 1.5, global_threshold)
            template, _, _ = ImageProcessing3D.create_in_shell(rotated_bin, (0, 2), penalty=100)
            
            res = tetris_obj.insert_molecule_3d(template, rotated, name, allowed_mask, current_target, box_size)
            if res == 'inserted':
                total_inserted += 1
                consecutive_failures = 0
                current_target = tetris_obj.all_coordinates[-1]
            else:
                consecutive_failures += 1
                current_target = _pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)

    final = np.copy(tetris_obj.output_volume)
    final[~allowed_mask] = 0.0
    combined = final + ((~allowed_mask).astype(np.float32) * (MEMBRANE_INTENSITY_SCALE * final.max()))
    lio.write_mrc(combined, str(Path(output_dir)/f"membrane_{membrane_id:02d}/tomo_den.mrc"), v_size=VOI_VSIZE)
    print(f"DONE: {total_inserted} inserted in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    insert_proteins_in_membrane(MEMBRANES_PATH / MEMBRANE_FILES[0], PROTEINS_LIST, OUT_DIR, 0)