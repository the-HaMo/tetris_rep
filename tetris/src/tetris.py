"""
Tetris Algorithm 3D 
"""

import time
import numpy as np
import os
from typing import Tuple, List
from image_processing_3d import ImageProcessing3D
from parser_3d import Parser3D


PROTEINS_LIST = [
    "in_10A/4v4r_10A.pns",
    "in_10A/3j9i_10A.pns",
    "in_10A/5mrc_10A.pns",
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
    # "in_10A/1bxn_10A.pns",
]

EXPORT_VTP = True
VTP_BASE_NAME = "tomo_009_poly"
VTP_ISO_LEVEL = None  # None -> usa percentil automático


class Tetris3D:
    """
    Algoritmo Tetris 3D - trabaja con volúmenes completos.
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int] = (256, 256, 256),
                 sigma: float = 1.5,
                 threshold: float = 50,
                 insertion_distances: Tuple[int, int] = (-2, 0)):
        
        self.dimensions = np.array(dimensions)
        self.sigma = sigma
        self.threshold = threshold
        self.insertion_distances = insertion_distances
        
        self.reset()
    
    def reset(self):
        """Reinicia el volumen de salida."""
        self.output_volume = np.zeros(self.dimensions, dtype=np.float32)
        self.insertion_labels = np.zeros(self.dimensions, dtype=np.int32)  # Volumen de labels
        self.all_coordinates = []
        self.all_molecule_types = []  # Tipo de cada molécula insertada
    
    def get_occupancy(self) -> float:
        """Calcula la ocupancia actual del volumen (porcentaje de vóxeles ocupados)."""
        binary = ImageProcessing3D.smooth_and_binarize(
            self.output_volume, self.sigma, self.threshold
        )
        occupied = np.sum(binary > 0)
        total = np.prod(self.dimensions)
        return occupied / total
    
    def place_molecule_3d(self, position: Tuple[int, int, int], molecule: np.ndarray, molecule_id: int = 0):
        """
        Coloca una molécula 3D en el volumen de salida.
        
        Args:
            position: Coordenadas (z, y, x) centrales
            molecule: Volumen 3D de la molécula
            molecule_id: ID de la molécula para el volumen de labels
        """
        size = molecule.shape
        half = np.array(size) // 2
        z, y, x = position
        
        # Calcular límites en el output
        z_start = max(0, z - half[0])
        z_end = min(self.dimensions[0], z + half[0])
        y_start = max(0, y - half[1])
        y_end = min(self.dimensions[1], y + half[1])
        x_start = max(0, x - half[2])
        x_end = min(self.dimensions[2], x + half[2])
        
        # Calcular límites en la molécula
        mol_z_start = half[0] - (z - z_start)
        mol_z_end = half[0] + (z_end - z)
        mol_y_start = half[1] - (y - y_start)
        mol_y_end = half[1] + (y_end - y)
        mol_x_start = half[2] - (x - x_start)
        mol_x_end = half[2] + (x_end - x)
        
        # Añadir molécula al output
        mol_region = molecule[mol_z_start:mol_z_end, mol_y_start:mol_y_end, mol_x_start:mol_x_end]
        self.output_volume[z_start:z_end, y_start:y_end, x_start:x_end] += mol_region
        
        # Actualizar labels donde la molécula tiene densidad significativa
        mol_mask = mol_region > (mol_region.max() * 0.3)  # Umbral del 30%
        self.insertion_labels[z_start:z_end, y_start:y_end, x_start:x_end][mol_mask] = molecule_id
    
    def insert_molecule_3d(self, molecule: np.ndarray, mol_name: str = "mol") -> bool:
        """
        Inserta una molécula 3D siguiendo el algoritmo Tetris.
        
        Args:
            molecule: Volumen 3D de la molécula
            mol_name: Nombre para logging
            
        Returns:
            True si se insertó, False si saturación
        """
        step = len(self.all_coordinates) + 1
        center = tuple(self.dimensions // 2)
        
        rotated, angles = ImageProcessing3D.randomly_rotate(molecule)
        box_size = max(rotated.shape)
        
        if len(self.all_coordinates) == 0:
            self.place_molecule_3d(center, rotated, molecule_id=step)
            self.all_coordinates.append(center)
            self.all_molecule_types.append(mol_name)
            print(f"  Paso {step}: {mol_name} insertada en centro {center}")
            return True
        
        rotated_binary = ImageProcessing3D.smooth_and_binarize(
            rotated, self.sigma, self.threshold
        )
        
        template, outer_layer, inner_layer = ImageProcessing3D.create_in_shell(
            rotated_binary, self.insertion_distances
        )
    
        output_binary = ImageProcessing3D.smooth_and_binarize(
            self.output_volume, self.sigma, self.threshold
        )
        
        cmap = ImageProcessing3D.correlate(output_binary, template, box_size)
        
        if cmap.max() <= 0:
            print(f"  Paso {step}: SATURACIÓN - No se puede insertar {mol_name}")
            return False
        
        # Paso 8: Insertar en posición óptima
        coord = ImageProcessing3D.find_maximum_position(cmap)
        self.place_molecule_3d(coord, rotated, molecule_id=step)
        self.all_coordinates.append(coord)
        self.all_molecule_types.append(mol_name)
        
        occupancy = self.get_occupancy()
        print(f"  Paso {step}: {mol_name} insertada en {coord} - Ocupancia: {occupancy*100:.1f}%")
        return True


def run_tetris_3d():
    """Ejecuta el algoritmo Tetris 3D."""
    
    # Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'data', 'data_generated', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar proteínas
    print(f"\nCargando {len(PROTEINS_LIST)} proteínas:")
    molecules = {}
    molecule_params = {}
    for protein_path in PROTEINS_LIST:
        full_path = os.path.join(data_dir, protein_path)
        if os.path.exists(full_path):
            name = os.path.basename(protein_path)
            try:
                volume, params = Parser3D.load_protein(full_path, data_dir)
                molecules[name] = volume
                molecule_params[name] = params
                occupancy = params.get('PMER_OCC', 'N/A')
                print(f"  {name}: {volume.shape} - Ocupancia objetivo: {occupancy}")
            except Exception as e:
                print(f"  {name}: Error - {e}")
        else:
            print(f"  {protein_path}: NO ENCONTRADO")
    
    if len(molecules) == 0:
        print("\n¡ERROR! No se encontraron proteínas.")
        return None
        
    # Crear Tetris 3D
    tetris = Tetris3D(
        dimensions=(300, 300, 250), # VOI SHAPE 
        sigma=1.5,
        threshold=50,
        insertion_distances=(-2, 0)
    )
    
    # Insertar moléculas
    mol_list = list(molecules.values())
    mol_names = list(molecules.keys())
    mol_occupancies = {}
    
    # Extraer ocupancias objetivo de cada proteína
    for name in mol_names:
        params = molecule_params.get(name, {})
        try:
            mol_occupancies[name] = float(params.get('PMER_OCC', 0.5))
        except ValueError:
            mol_occupancies[name] = 0.5  # Default si no se puede leer
    
    # Usar la ocupancia máxima como objetivo global
    target_occupancy = max(mol_occupancies.values())
    print(f"\n Objetivo: Ocupancia {target_occupancy*100:.1f}% o saturación")
    
    inserted = 0
    inserted_per_protein = {name: 0 for name in mol_names}
    
    # Bucle infinito - solo se detiene por ocupancia o saturación
    while True:
        mol_idx = inserted % len(mol_list)
        mol_name = mol_names[mol_idx]
        
        success = tetris.insert_molecule_3d(
            mol_list[mol_idx],
            mol_name=mol_name
        )
        if success:
            inserted += 1
            inserted_per_protein[mol_name] += 1
            
            # Verificar si se alcanzó la ocupancia objetivo
            current_occupancy = tetris.get_occupancy()
            if current_occupancy >= target_occupancy:
                print(f"\n✓ OCUPANCIA OBJETIVO alcanzada: {current_occupancy*100:.1f}% >= {target_occupancy*100:.1f}%")
                print(f"  Total de moléculas insertadas: {inserted}")
                print(f"  Distribución por proteína:")
                for prot_name, count in inserted_per_protein.items():
                    print(f"    {prot_name}: {count}")
                break
        else:
            # Saturación - no caben más moléculas
            current_occupancy = tetris.get_occupancy()
            print(f"\n✓ SATURACIÓN alcanzada - No caben más moléculas")
            print(f"  Ocupancia final: {current_occupancy*100:.1f}%")
            print(f"  Total de moléculas insertadas: {inserted}")
            print(f"  Distribución por proteína:")
            for prot_name, count in inserted_per_protein.items():
                print(f"    {prot_name}: {count}")
            break
    
    # Guardar resultado
    output_path = os.path.join(output_dir, 'tetris_3d_output.mrc')
    Parser3D.save_output_files(
        output_volume=tetris.output_volume,
        insertion_labels=tetris.insertion_labels,
        coordinates=tetris.all_coordinates,
        molecule_types=tetris.all_molecule_types,
        filepath=output_path,
        voxel_size=10.0
    )

    if EXPORT_VTP:
        Parser3D.save_vtp_files(
            output_volume=tetris.output_volume,
            insertion_labels=tetris.insertion_labels,
            output_dir=output_dir,
            base_name=VTP_BASE_NAME,
            voxel_size=10.0,
            iso_level=VTP_ISO_LEVEL,
            sigma=tetris.sigma,
            threshold=tetris.threshold
        )
    
    return tetris


if __name__ == '__main__':
    start_time = time.time()
    
    tetris = run_tetris_3d()
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    print(f"\nTiempo total de ejecución: {minutes} min {seconds:.2f} s")
