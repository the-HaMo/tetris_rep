"""
Algoritmo Tetris 2D

"""

import mrcfile
import numpy as np
import os
from typing import Tuple, List

from parse import ProteinParser
from image_processing import ImageProcessing


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

MAX_ATTEMPTS = 10000  # Límite de seguridad para inserciones
CANVAS_MULTIPLIER = 8  # Factor para tamaño del canvas

class TetrisAnimation2D:
    """
    Algoritmo Tetris 2D que genera una animación como stack 3D.
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int] = (256, 256),
                 sigma: float = 1.5,
                 threshold: float = 50,
                 insertion_distances: Tuple[int, int] = (-2, 0)):
        
        self.dimensions = np.array(dimensions)
        self.sigma = sigma
        self.threshold = threshold
        self.insertion_distances = insertion_distances
        
        self.reset()
        
    def reset(self):
        self.output_image = np.zeros(self.dimensions, dtype=np.float32)
        self.all_coordinates = []
        self.frames = []  # Lista de frames para la animación
        self.frame_labels = []  # Etiquetas para cada frame
        self._current_frontier = None  # Cache de la frontera actual
        self._output_binary = None  # Cache del output binarizado
    
    def place_molecule(self, position: Tuple[int, int], molecule: np.ndarray):
        """Añade una molécula al output en la posición dada."""
        size = molecule.shape[0]
        half = size // 2
        x, y = position
        
        x_start = max(0, x - half)
        x_end = min(self.dimensions[0], x + half)
        y_start = max(0, y - half)
        y_end = min(self.dimensions[1], y + half)
        
        mol_x_start = half - (x - x_start)
        mol_x_end = half + (x_end - x)
        mol_y_start = half - (y - y_start)
        mol_y_end = half + (y_end - y)
        
        self.output_image[x_start:x_end, y_start:y_end] += molecule[mol_x_start:mol_x_end, mol_y_start:mol_y_end]
    
    def add_frame(self, image: np.ndarray, label: str):
        """Añade un frame a la animación."""
        self.frames.append(image.copy())
        self.frame_labels.append(label)
    
    def _create_frame_with_frontier(self, mol_name: str, coord: Tuple[int, int], 
                                    step: int, box_size: int, use_local: bool = True):
        """
        Crea un frame con el output actual y la frontera resaltada.
        
        Args:
            mol_name: Nombre de la molécula
            coord: Coordenada de inserción
            step: Número de paso
            box_size: Tamaño de la caja de la molécula
            use_local: Si True, actualiza frontera localmente (más eficiente)
        """
        # Actualizar output binarizado
        self._output_binary = ImageProcessing.smooth_and_binarize(
            self.output_image, self.sigma, self.threshold
        )
        
        if use_local and self._current_frontier is not None and step > 1:
            # Actualización LOCAL - solo la región afectada
            self._current_frontier = ImageProcessing.update_frontier_local(
                self._current_frontier,
                self._output_binary,
                coord,
                box_size
            )
        else:
            # Actualización GLOBAL - primera vez o forzado
            self._current_frontier = ImageProcessing.compute_frontier(self._output_binary)
        
        # Crear frame con frontera resaltada
        frame = self.output_image.copy()
        max_val = frame.max() if frame.max() > 0 else 1000
        frame[self._current_frontier > 0] = max_val * 1.5
        
        label = f"Paso {step}: {mol_name} en {coord}"
        self.add_frame(frame, label)
    
    def insert_molecule(self, molecule: np.ndarray, mol_name: str = "mol", 
                       mode: str = 'output', use_local_frontier: bool = True) -> bool:
        """
        Insert a molecule following the Tetris algorithm.
        
        Flujo del diagrama:
        1. Randomly rotate molecule
        2. First iteration? → Place in center
        3. Smooth and binarize → Dilate → Subtract (In-shell)
        4. Correlate → C-map
        5. C-map negative? → End (saturación)
        6. Add molecule at maximum → Update output → Iterate
        """
        step = len(self.all_coordinates) + 1
        center = tuple(self.dimensions // 2)
        
        # Paso 1: Randomly rotate one molecule
        rotated, angle = ImageProcessing.randomly_rotate(molecule)
        box_size = rotated.shape[0]
        
        # Paso 2: First iteration?
        if len(self.all_coordinates) == 0:
            # YES → Place it in the center of an empty sample
            self.place_molecule(center, rotated)
            self.all_coordinates.append(center)
            self._create_frame_with_frontier(mol_name, center, step, box_size, use_local=False)
            print(f"  Paso {step}: {mol_name} insertada en centro {center}")
            return True
        
        # NO → Continuar con el algoritmo completo
        
        # Paso 3: Smooth and binarize (molécula)
        rotated_binary = ImageProcessing.smooth_and_binarize(
            rotated, self.sigma, self.threshold
        )
        
        # Paso 4: Dilate → Subtract → In-shell template
        template, outer_layer, inner_layer = ImageProcessing.create_in_shell(
            rotated_binary, self.insertion_distances
        )
        
        # Paso 5: Smooth and binarize (output actual)
        output_binary = ImageProcessing.smooth_and_binarize(
            self.output_image, self.sigma, self.threshold
        )
        
        # Paso 6: Correlate → C-map
        cmap = ImageProcessing.correlate(output_binary, template, box_size)
        
        # Paso 7: C-map is negative?
        if ImageProcessing.is_cmap_negative(cmap):
            # YES → End (saturación)
            print(f"  Paso {step}: SATURACIÓN - No se puede insertar {mol_name}")
            return False
        
        # NO → Add the molecule at maximum value entry of c-map
        coord = ImageProcessing.find_maximum_position(cmap)
        
        # Update output
        self.place_molecule(coord, rotated)
        self.all_coordinates.append(coord)
        
        # Crear frame para animación (con actualización LOCAL de frontera)
        self._create_frame_with_frontier(mol_name, coord, step, box_size, use_local=use_local_frontier)
        print(f"  Paso {step}: {mol_name} insertada en {coord}")
        
        return True  # Iterate
    
    def save_animation(self, filepath: str, voxel_size: float = 10.0):
        """Guarda todos los frames como un stack 3D."""
        if len(self.frames) == 0:
            print("No hay frames para guardar")
            return
        
        # Crear stack 3D (Z, Y, X)
        stack = np.array(self.frames, dtype=np.float32)
        
        with mrcfile.new(filepath, overwrite=True) as mrc:
            mrc.set_data(stack)
            mrc.voxel_size = voxel_size
        
        print(f"\n Animación guardada: {filepath}")
        print(f"  Frames: {len(self.frames)}")
        print(f"  Dimensiones: {stack.shape}")
        
        # Guardar etiquetas
        labels_path = filepath.replace('.mrc', '_labels.txt')
        with open(labels_path, 'w') as f:
            for i, label in enumerate(self.frame_labels):
                f.write(f"Z={i}: {label}\n")
        print(f"  Etiquetas: {labels_path}")
        
        return filepath


def run_animation():
    """Genera la animación del Tetris."""
    print("=" * 60)
    print("TETRIS - ANIMACIÓN DE INSERCIÓN DE PROTEÍNAS")
    print("=" * 60)
    
    # Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'data', 'data_generated', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar proteínas desde PROTEINS_LIST
    print(f"\nCargando {len(PROTEINS_LIST)} proteínas:")
    molecules = {}
    for protein_path in PROTEINS_LIST:
        full_path = os.path.join(data_dir, protein_path)
        if os.path.exists(full_path):
            name = os.path.basename(protein_path)
            try:
                molecules[name] = ProteinParser.load_protein(full_path, data_dir)
                print(f"{name}: {molecules[name].shape}")
            except Exception as e:
                print(f"{name}: Error - {e}")
        else:
            print(f"{protein_path}: NO ENCONTRADO")
    
    if len(molecules) == 0:
        print("\n¡ERROR! No se encontraron proteínas. Verifica PROTEINS_LIST.")
        return None
    
    # Calcular tamaño del canvas
    max_size = max(m.shape[0] for m in molecules.values())
    canvas_size = max_size * CANVAS_MULTIPLIER  # Más grande para más inserciones
    
    print(f"\nCanvas: {canvas_size} x {canvas_size}")
    
    # Crear animador
    tetris = TetrisAnimation2D(
        dimensions=(canvas_size, canvas_size),
        sigma=1.5,
        threshold=50,
        insertion_distances=(-2, 0)
    )
    
    # Insertar moléculas (alternando tipos)
    mol_list = list(molecules.values())
    mol_names = list(molecules.keys())
    
    print(f"\nInsertando moléculas hasta saturación...")
    print("-" * 40)
    
    inserted = 0
    for i in range(MAX_ATTEMPTS):
        mol_idx = i % len(mol_list)
        success = tetris.insert_molecule(
            mol_list[mol_idx],
            mol_name=mol_names[mol_idx],
            mode='output'
        )
        if success:
            inserted += 1
        else:
            print(f"\nSATURACIÓN alcanzada después de {inserted} moléculas")
            break
    
    # Guardar animación
    output_path = os.path.join(output_dir, 'tetris_insertion.mrc')
    tetris.save_animation(output_path)
    
    return tetris


if __name__ == '__main__':
    tetris = run_animation()
