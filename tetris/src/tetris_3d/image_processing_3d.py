"""
image_processing_3d.py
==============================================
Contiene las operaciones esenciales para el flujo de trabajo de Tetris 3D.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage.morphology import ball
from typing import Tuple
import multiprocessing
import scipy.fft

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
    print(f"[OPT] PyFFTW activado usando {pyfftw.config.NUM_THREADS} hilos.")
except ImportError:
    print("[WARN] PyFFTW no instalado. Correlación 3D usará 1 solo hilo.")

# GPU detection 
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi_gpu
    xp = cp
    ndi = ndi_gpu
    GPU_AVAILABLE = True
except ImportError:
    import scipy.ndimage as ndi_cpu
    xp = np
    ndi = ndi_cpu
    GPU_AVAILABLE = False

class ImageProcessing3D:
    """
    Operaciones de procesamiento de volúmenes 3D optimizadas.
    """
    
    @staticmethod
    def randomly_rotate(data: np.ndarray, angles: Tuple[float, float, float] = None):
        if angles is None:
            angles = np.random.uniform(0, 360, size=3)
        
        rotation = R.from_euler('zyx', angles, degrees=True)
        matrix = rotation.as_matrix()
        center = np.array(data.shape) / 2.0
        offset = center - np.dot(matrix, center)

        # Si hay GPU, convertimos los parámetros a CuPy para evitar el ValueError
        device_data = xp.asarray(data)
        device_matrix = xp.asarray(matrix)
        device_offset = xp.asarray(offset)

        # Usamos 'ndi' que apunta al backend detectado (SciPy o CuPy)
        rotated = ndi.affine_transform(
            device_data, device_matrix, offset=device_offset, 
            order=1, mode='constant', cval=0.0
        )
        return rotated, tuple(angles)
    
    @staticmethod
    def smooth_and_binarize(data, sigma: float = 1.5, threshold: float = 50):
        if sigma > 0:
            data = ndi.gaussian_filter(xp.asarray(data), sigma)
        return (data > threshold).astype(xp.float32)

    @staticmethod
    def dilate(binary_volume, distance: int):
        """
        Dilatación 3D necesaria para generar las capas del template.
        Usa el backend activo para evitar mezclar NumPy con CuPy.
        """
        device_binary = xp.asarray(binary_volume)
        structure = xp.asarray(ball(max(1, distance)))
        if distance == 0:
            return ndi.binary_closing(device_binary, structure=structure).astype(xp.float32)
        return ndi.binary_dilation(device_binary, structure=structure).astype(xp.float32)
    
    @staticmethod
    def subtract(outer_layer: np.ndarray, inner_layer: np.ndarray, 
                 penalty: float = 100) -> np.ndarray:
        """
        Crea el molde restando el núcleo (con penalización) de la capa externa. 
        """
        template = np.zeros_like(outer_layer, dtype=np.float32)
        # Cáscara: puntos donde queremos que haya densidad (correlación positiva)
        shell_mask = np.logical_and(outer_layer > 0, ~(inner_layer > 0))
        template[shell_mask] = 1.0
        # Núcleo: puntos de choque prohibido (penalización negativa)
        template[inner_layer > 0] = -penalty
        return template

    @staticmethod
    def create_in_shell(binary_vol: np.ndarray, 
                        insertion_distances: Tuple[int, int] = (0, 2),
                        penalty: float = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera el template "in-shell" completo combinando dilatación y resta. 
        """
        inner_dist, outer_dist = insertion_distances
        outer_layer = ImageProcessing3D.dilate(binary_vol, outer_dist)
        inner_layer = ImageProcessing3D.dilate(binary_vol, inner_dist)
        template = ImageProcessing3D.subtract(outer_layer, inner_layer, penalty)
        
        return template, outer_layer, inner_layer