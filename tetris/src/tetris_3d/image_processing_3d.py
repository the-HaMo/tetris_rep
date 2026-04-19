"""
image_processing_3d.py
==============================================
Contiene las operaciones esenciales para el flujo de trabajo de Tetris 3D.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, affine_transform
from scipy.spatial.transform import Rotation as R
from skimage.morphology import ball
from typing import Tuple
import multiprocessing
import scipy.fft

try:
    import pyfftw
    # Configurar para usar todos los núcleos disponibles (8 CPUs) 
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
    print(f"[OPT] PyFFTW activado usando {pyfftw.config.NUM_THREADS} hilos.")
except ImportError:
    print("[WARN] PyFFTW no instalado. Correlación 3D usará 1 solo hilo.")

class ImageProcessing3D:
    """
    Operaciones de procesamiento de volúmenes 3D optimizadas.
    """
    
    @staticmethod
    def randomly_rotate(data: np.ndarray, 
                       angles: Tuple[float, float, float] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Rota un volumen 3D en una sola pasada usando una matriz de transformación afín.
        Es mucho más eficiente que rotar eje por eje. 
        """
        if angles is None:
            angles = np.random.uniform(0, 360, size=3)
        
        # Crear matriz de rotación combinada (Z-Y-X)
        rotation = R.from_euler('zyx', angles, degrees=True)
        matrix = rotation.as_matrix()
        
        # Calcular centro para rotar sobre el propio eje del volumen
        center = np.array(data.shape) / 2.0
        offset = center - np.dot(matrix, center)
        
        # Aplicar transformación en una sola pasada
        rotated = affine_transform(
            data, 
            matrix, 
            offset=offset, 
            order=1, 
            mode='constant', 
            cval=0.0
        )
        return rotated, tuple(angles)
    
    @staticmethod
    def smooth_and_binarize(data: np.ndarray, sigma: float = 1.5, 
                           threshold: float = 50) -> np.ndarray:
        """
        Limpia el volumen rotado aplicando un filtro Gaussiano y binarizando. 
        """
        if sigma > 0:
            data = gaussian_filter(data, sigma)
        return (data > threshold).astype(np.float32)

    @staticmethod
    def dilate(binary_volume: np.ndarray, distance: int) -> np.ndarray:
        """
        Dilatación 3D necesaria para generar las capas del template. 
        """
        if distance == 0:
            return binary_closing(binary_volume, ball(1)).astype(np.float32)
        return binary_dilation(binary_volume, ball(distance)).astype(np.float32)
    
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