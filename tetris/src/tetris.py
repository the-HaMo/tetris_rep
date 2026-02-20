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

VOI_SHAPE = (300, 300, 250)
VOXEL_SIZE = 10.0  # nm
FACTOR_DOWNSAMPLE = 2 

EXPORT_VTP = True
VTP_BASE_NAME = "tomo_009_poly"
VTP_ISO_LEVEL = None  # None -> usa percentil automático

# OPTIMIZACIONES
USE_OPTIMIZATIONS = True  # Activar optimizaciones (ROI + cache local)
USE_DOWNSAMPLING = True  # ULTRA: Correlación downsampled (4-6x más rápido)
DEBUG_OPTIMIZATIONS = True  # Mostrar info de ROI en cada paso (ver impacto real)


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
        # OPTIMIZACIÓN: Cache para evitar recalcular todo cada vez
        self._output_binary = None  # Cache del volumen binarizado
        self._current_frontier = None  # Cache de la frontera
        self._step_count = 0  # Contador para gaussian cada 100 pasos
    
    def get_occupancy(self) -> float:
        """Calcula la ocupancia actual del volumen (porcentaje de vóxeles ocupados)."""
        # Usar cache si disponible, sino recalcular con gaussian
        if self._output_binary is None:
            binary = ImageProcessing3D.smooth_and_binarize(
                self.output_volume, self.sigma, self.threshold
            )
        else:
            binary = self._output_binary
        occupied = np.sum(binary > 0)
        total = np.prod(self.dimensions)
        return occupied / total
    
    def place_molecule_3d(self, position: Tuple[int, int, int], molecule: np.ndarray, molecule_id: int = 0) -> bool:
        """
        Coloca una molécula 3D en el volumen de salida.
        Verifica ANTES de insertar que no haya overlaps.
        
        Args:
            position: Coordenadas (z, y, x) centrales
            molecule: Volumen 3D de la molécula
            molecule_id: ID de la molécula para el volumen de labels
            
        Returns:
            True si se insertó exitosamente, False si hay overlap
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
        
        # Extraer regiones
        mol_region = molecule[mol_z_start:mol_z_end, mol_y_start:mol_y_end, mol_x_start:mol_x_end]
        output_region = self.output_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # VERIFICACIÓN CRÍTICA: Detectar overlaps
        # Donde la molécula es densa (>30%), no debería haber densidad preexistente
        mol_mask = mol_region > (mol_region.max() * 0.3)  # Vóxeles significativos de molécula
        
        if (output_region[mol_mask] > 0).any():
            # HAY OVERLAP - molécula densa se superpone con densidad existente
            return False
        
        # Sin overlap - insertar
        self.output_volume[z_start:z_end, y_start:y_end, x_start:x_end] += mol_region
        
        # Actualizar labels donde la molécula tiene densidad significativa
        self.insertion_labels[z_start:z_end, y_start:y_end, x_start:x_end][mol_mask] = molecule_id
        
        return True
    
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
            
            # Inicializar cache para primera molécula
            self._output_binary = ImageProcessing3D.smooth_and_binarize(
                self.output_volume, self.sigma, self.threshold
            )
            self._current_frontier = ImageProcessing3D.compute_frontier(self._output_binary)
            
            print(f"  Paso {step}: {mol_name} insertada en centro {center}")
            return True
        
        rotated_binary = ImageProcessing3D.smooth_and_binarize(
            rotated, self.sigma, self.threshold
        )
        
        template, outer_layer, inner_layer = ImageProcessing3D.create_in_shell(
            rotated_binary, self.insertion_distances
        )
    
        # ESTRATEGIA HÍBRIDA: threshold rápido + gaussian cada 100 pasos
        self._step_count += 1
        
        if self._step_count % 100 == 0:
            # Cada 100 inserciones: gaussian para corregir drift y evitar overlaps
            self._output_binary = ImageProcessing3D.smooth_and_binarize(
                self.output_volume, self.sigma, self.threshold
            )
            self._current_frontier = ImageProcessing3D.compute_frontier(self._output_binary)
        else:
            # 99 de 100: threshold ultra-rápido
            self._output_binary = ImageProcessing3D.threshold_binarize(
                self.output_volume, self.threshold
            )
            # Frontera se reutiliza del último gaussian (no recalcular)
        
        # OPTIMIZACIÓN 1: Correlación con ROI (solo en región de frontera)
        # OPTIMIZACIÓN ULTRA: Downsampling (correlación en escala reducida)
        if USE_DOWNSAMPLING:
            # Método más rápido: correlación downsampled + refinamiento local
            if DEBUG_OPTIMIZATIONS and step % 2 == 0:
                bounds = ImageProcessing3D.get_frontier_roi_bounds(
                    self._current_frontier, box_size, expansion_factor=1.1
                )
                roi_vol = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])
                total_vol = self._output_binary.shape[0] * self._output_binary.shape[1] * self._output_binary.shape[2]
                roi_pct = 100.0 * roi_vol / total_vol
                print(f"    [PASO {step}] ROI: {roi_pct:.1f}% del volumen")
            
            cmap = ImageProcessing3D.correlate_downsampled(
                self._output_binary, template, box_size,
                downsample_factor=FACTOR_DOWNSAMPLE,  # Factor 2 = 8x más rápido
                refine=True,          # Refinar en región pequeña
                refine_size=25        # ±50 voxeles: suficiente para templates grandes (72x72x72)
            )
            
            if DEBUG_OPTIMIZATIONS and step % 100 == 0:
                print(f"    [PASO {step}] DOWNSAMPLING: factor 2 (8x velocidad)")
                
        elif USE_OPTIMIZATIONS:
            # Método anterior: ROI basado en frontera (no tan efectivo)
            cmap = ImageProcessing3D.correlate_roi(
                self._output_binary, template, box_size, 
                frontier=self._current_frontier,
                expansion_factor=1.1  # ROI más agresivo que antes (era 1.5)
            )
            
            # Debug cada 100 pasos para no saturar output
            if DEBUG_OPTIMIZATIONS and self._current_frontier is not None and step % 100 == 0:
                roi_bounds = ImageProcessing3D.get_frontier_roi_bounds(
                    self._current_frontier, box_size, expansion_factor=1.1
                )
                if roi_bounds:
                    z_start, z_end, y_start, y_end, x_start, x_end = roi_bounds
                    roi_volume = (z_end - z_start) * (y_end - y_start) * (x_end - x_start)
                    total_volume = np.prod(self.dimensions)
                    roi_fraction = roi_volume / total_volume
                    print(f"    [PASO {step}] ROI: {roi_fraction*100:.1f}% del volumen total")
        else:
            # Sin optimización: correlación completa
            cmap = ImageProcessing3D.correlate(self._output_binary, template, box_size)
        
        if cmap.max() <= 0:
            print(f"  Paso {step}: SATURACIÓN - No se puede insertar {mol_name}")
            return False
        
        # Paso 8: Insertar en posición óptima
        coord = ImageProcessing3D.find_maximum_position(cmap)
        
        # Verificar overlaps antes de insertar
        inserted = self.place_molecule_3d(coord, rotated, molecule_id=step)
        if not inserted:
            # Overlap detectado en esta posición - rechazar inserción
            return False
        
        self.all_coordinates.append(coord)
        self.all_molecule_types.append(mol_name)
        
        # CRÍTICO: Actualizar binary cache después de insertar
        self._output_binary = ImageProcessing3D.threshold_binarize(
            self.output_volume, self.threshold
        )
        
        # OPTIMIZACIÓN: Actualizar frontera LOCALMENTE en vez de globalmente
        # (Solo recalcula región afectada - 10x más rápido que frontera global)
        if self._current_frontier is None:
            # Primera molécula: calcular frontera global
            self._current_frontier = ImageProcessing3D.compute_frontier(self._output_binary)
        else:
            # Siguientes moléculas: actualizar solo la región local donde se insertó
            self._current_frontier = ImageProcessing3D.update_frontier_local(
                self._current_frontier,
                self._output_binary,
                coord,
                box_size,
                radius=3,
                margin=5
            )
        
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
        dimensions=VOI_SHAPE,
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
    last_occ_check = 0
    
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
            
            # Verificar ocupancia cada 200 inserciones (no cada una - mucho más rápido)
            if inserted - last_occ_check >= 200:
                current_occupancy = tetris.get_occupancy()
                print(f"  {inserted} proteínas - Ocupancia: {current_occupancy*100:.1f}%")
                last_occ_check = inserted
                
                if current_occupancy >= target_occupancy:
                    print(f"\n✓ OCUPANCIA OBJETIVO: {current_occupancy*100:.1f}%")
                    print(f"  Proteínas insertadas: {inserted}")
                    for prot_name, count in inserted_per_protein.items():
                        print(f"    {prot_name}: {count}")
                    break
        else:
            # Saturación - no caben más moléculas
            current_occupancy = tetris.get_occupancy()
            print(f"\n✓ SATURACIÓN - Ocupancia final: {current_occupancy*100:.1f}%")
            print(f"  Proteínas insertadas: {inserted}")
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
            voxel_size=VOXEL_SIZE,
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
