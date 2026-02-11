# -*- coding: utf-8 -*-
"""
Procesamiento de Volúmenes 3D para el Algoritmo Tetris
=======================================================

Operaciones 3D basadas en el diagrama del paper adaptadas a volúmenes:
- Smooth and Binarize (3D)
- Dilate (3D)
- Subtract (In-shell 3D)
- Correlate (3D)
- Frontier computation (3D)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_erosion, rotate
from scipy import signal
from skimage.morphology import ball
from typing import Tuple


class ImageProcessing3D:
    """
    Clase con operaciones de procesamiento de volúmenes 3D para el algoritmo Tetris.
    Extiende las operaciones 2D al espacio tridimensional.
    """
    
    @staticmethod
    def randomly_rotate(data: np.ndarray, 
                       angles: Tuple[float, float, float] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Paso: Randomly rotate one molecule (3D)
        
        Rota un volumen 3D aleatoriamente en los 3 ejes.
        
        Args:
            data: Volumen 3D (Z, Y, X)
            angles: (angle_z, angle_y, angle_x) o None para aleatorio
            
        Returns:
            Tupla (volumen rotado, ángulos usados)
        """
        if angles is None:
            angles = (
                np.random.uniform(0, 360),
                np.random.uniform(0, 360),
                np.random.uniform(0, 360)
            )
        
        rotated = data.copy()
        # Rotar en eje Z (plano XY)
        rotated = rotate(rotated, angles[0], axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
        # Rotar en eje Y (plano XZ)
        rotated = rotate(rotated, angles[1], axes=(0, 2), reshape=False, order=1, mode='constant', cval=0)
        # Rotar en eje X (plano YZ)
        rotated = rotate(rotated, angles[2], axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
        
        return rotated, angles
    
    @staticmethod
    def smooth_and_binarize(data: np.ndarray, sigma: float = 1.5, 
                           threshold: float = 50) -> np.ndarray:
        """
        Paso: Smooth and binarize (3D)
        
        Aplica filtro Gaussiano y binariza el volumen.
        
        Args:
            data: Volumen 3D de entrada
            sigma: Desviación estándar del filtro Gaussiano
            threshold: Umbral base para binarización
            
        Returns:
            Volumen binarizado (0 o 1)
        """
        smoothed = gaussian_filter(data, sigma)
        if np.max(smoothed) > 0:
            thresh = max(threshold, np.percentile(smoothed[smoothed > 0], 30))
        else:
            thresh = threshold
        return (smoothed > thresh).astype(np.float32)
    
    @staticmethod
    def dilate(binary_volume: np.ndarray, distance: int) -> np.ndarray:
        """
        Paso: Dilate (3D)
        
        Expande (dilata) o contrae (erosiona) un volumen binario.
        
        Args:
            binary_volume: Volumen binario
            distance: Distancia de dilatación (+ dilata, - erosiona, 0 closing)
            
        Returns:
            Volumen dilatado/erosionado
        """
        if distance == 0:
            return binary_closing(binary_volume, ball(1)).astype(np.float32)
        elif distance < 0:
            closed = binary_closing(binary_volume, ball(1))
            return binary_erosion(closed, ball(-distance)).astype(np.float32)
        else:
            return binary_dilation(binary_volume, ball(distance)).astype(np.float32)
    
    @staticmethod
    def subtract(outer_layer: np.ndarray, inner_layer: np.ndarray, 
                 penalty: float = 100) -> np.ndarray:
        """
        Paso: Subtract (crear In-shell template 3D)
        
        Crea el template restando la capa interna (con penalización) de la externa.
        
        Args:
            outer_layer: Capa externa (dilatada)
            inner_layer: Capa interna (para penalización)
            penalty: Factor de penalización para solapamiento
            
        Returns:
            Template = outer - penalty * inner
        """
        return outer_layer.astype(np.float32) - penalty * inner_layer.astype(np.float32)
    
    @staticmethod
    def create_in_shell(binary_vol: np.ndarray, 
                        insertion_distances: Tuple[int, int] = (-2, 0),
                        penalty: float = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pasos combinados: Smooth, Dilate, Subtract (3D)
        
        Crea el template "in-shell" completo en 3D.
        
        Args:
            binary_vol: Volumen binarizado
            insertion_distances: (inner_dist, outer_dist)
            penalty: Factor de penalización
            
        Returns:
            Tupla (template, outer_layer, inner_layer)
        """
        inner_dist, outer_dist = insertion_distances
        
        # Dilate para capa externa
        outer_layer = ImageProcessing3D.dilate(binary_vol, outer_dist)
        
        # Dilate para capa interna
        inner_layer = ImageProcessing3D.dilate(binary_vol, inner_dist)
        
        # Subtract
        template = ImageProcessing3D.subtract(outer_layer, inner_layer, penalty)
        
        return template, outer_layer, inner_layer
    
    @staticmethod
    def correlate(output_binary: np.ndarray, template: np.ndarray, 
                  box_size: int) -> np.ndarray:
        """
        Paso: Correlate, C-map (3D)
        
        Calcula el mapa de correlación entre el output y el template en 3D.
        
        Args:
            output_binary: Output actual binarizado (volumen 3D)
            template: Template in-shell (volumen 3D)
            box_size: Tamaño de la caja de la molécula (para enmascarar bordes)
            
        Returns:
            C-map 3D (mapa de correlación con bordes enmascarados)
        """
        cmap = signal.correlate(output_binary, template, 'same', 'fft')
        
        # Enmascarar bordes en 3D
        half = box_size // 2
        cmap[:half+1, :, :] = 0
        cmap[:, :half+1, :] = 0
        cmap[:, :, :half+1] = 0
        cmap[-half-1:, :, :] = 0
        cmap[:, -half-1:, :] = 0
        cmap[:, :, -half-1:] = 0
        
        return cmap
    
    @staticmethod
    def is_cmap_negative(cmap: np.ndarray) -> bool:
        """
        Paso: C-map is negative? (3D)
        
        Verifica si el máximo del C-map es negativo o cero (saturación).
        
        Args:
            cmap: Mapa de correlación 3D
            
        Returns:
            True si saturación (no hay posición válida), False si hay espacio
        """
        return cmap.max() <= 0
    
    @staticmethod
    def find_maximum_position(cmap: np.ndarray) -> Tuple[int, int, int]:
        """
        Paso: Find maximum value entry of c-map (3D)
        
        Encuentra la posición del máximo en el C-map 3D.
        
        Args:
            cmap: Mapa de correlación 3D
            
        Returns:
            Coordenadas (z, y, x) del máximo
        """
        return np.unravel_index(cmap.argmax(), cmap.shape)
    
    @staticmethod
    def compute_frontier(binary_volume: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Calcula la frontera (borde exterior) de un volumen binario 3D (GLOBAL).
        
        La frontera es la región que rodea el volumen ocupado, útil para
        identificar dónde se pueden insertar nuevas moléculas.
        
        Args:
            binary_volume: Volumen binario 3D
            radius: Radio de dilatación para la frontera
            
        Returns:
            Volumen 3D de la frontera
        """
        if binary_volume.max() == 0:
            return np.zeros_like(binary_volume)
        
        dilated = binary_dilation(binary_volume, ball(radius))
        frontier = dilated.astype(np.float32) - binary_volume.astype(np.float32)
        
        return frontier
    
    @staticmethod
    def compute_frontier_local(binary_volume: np.ndarray, 
                               position: Tuple[int, int, int],
                               box_size: int,
                               radius: int = 3,
                               margin: int = 5) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
        """
        Calcula la frontera solo en la región afectada por una inserción 3D (LOCAL).
        
        Más eficiente que recalcular todo el volumen.
        
        Args:
            binary_volume: Volumen binario completo
            position: Posición (z, y, x) donde se insertó la molécula
            box_size: Tamaño de la caja de la molécula
            radius: Radio de dilatación para la frontera
            margin: Margen extra alrededor de la región
            
        Returns:
            Tupla (frontera_local, (z_start, z_end, y_start, y_end, x_start, x_end))
        """
        z, y, x = position
        half = box_size // 2
        
        # Calcular región de interés con margen
        z_start = max(0, z - half - radius - margin)
        z_end = min(binary_volume.shape[0], z + half + radius + margin)
        y_start = max(0, y - half - radius - margin)
        y_end = min(binary_volume.shape[1], y + half + radius + margin)
        x_start = max(0, x - half - radius - margin)
        x_end = min(binary_volume.shape[2], x + half + radius + margin)
        
        # Extraer región local
        local_region = binary_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if local_region.max() == 0:
            return np.zeros_like(local_region), (z_start, z_end, y_start, y_end, x_start, x_end)
        
        # Calcular frontera solo en la región local
        dilated = binary_dilation(local_region, ball(radius))
        frontier_local = dilated.astype(np.float32) - local_region.astype(np.float32)
        
        return frontier_local, (z_start, z_end, y_start, y_end, x_start, x_end)
    
    @staticmethod
    def update_frontier_local(current_frontier: np.ndarray,
                              binary_volume: np.ndarray,
                              position: Tuple[int, int, int],
                              box_size: int,
                              radius: int = 3,
                              margin: int = 5) -> np.ndarray:
        """
        Actualiza la frontera existente solo en la región afectada 3D (LOCAL).
        
        En lugar de recalcular toda la frontera, solo actualiza la zona
        donde se insertó la nueva molécula.
        
        Args:
            current_frontier: Frontera actual (se modifica in-place)
            binary_volume: Volumen binario actualizado
            position: Posición (z, y, x) donde se insertó la molécula
            box_size: Tamaño de la caja de la molécula
            radius: Radio de dilatación
            margin: Margen extra
            
        Returns:
            Frontera actualizada
        """
        # Calcular frontera local
        frontier_local, (z_start, z_end, y_start, y_end, x_start, x_end) = ImageProcessing3D.compute_frontier_local(
            binary_volume, position, box_size, radius, margin
        )
        
        # Actualizar solo la región afectada
        updated_frontier = current_frontier.copy()
        updated_frontier[z_start:z_end, y_start:y_end, x_start:x_end] = frontier_local
        
        return updated_frontier
