# -*- coding: utf-8 -*-
"""
Procesamiento de Imágenes para el Algoritmo Tetris
==================================================

Operaciones basadas en el diagrama del paper:
- Smooth and Binarize
- Dilate
- Subtract (In-shell)
- Correlate
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_erosion, rotate
from scipy import signal
from skimage.morphology import disk
from typing import Tuple


class ImageProcessing:
    """
    Clase con operaciones de procesamiento de imagen para el algoritmo Tetris.
    Sigue el flujo del diagrama del paper.
    """
    
    @staticmethod
    def randomly_rotate(data: np.ndarray, angle: float = None) -> Tuple[np.ndarray, float]:
        """
        Paso: Randomly rotate one molecule
        
        Args:
            data: Imagen 2D de la molécula
            angle: Ángulo de rotación (si None, genera uno aleatorio)
            
        Returns:
            Tupla (imagen rotada, ángulo usado)
        """
        if angle is None:
            angle = np.random.uniform(0, 360)
        rotated = rotate(data, angle, reshape=False, order=2, mode='constant', cval=0)
        return rotated, angle
    
    @staticmethod
    def smooth_and_binarize(data: np.ndarray, sigma: float = 1.5, 
                           threshold: float = 50) -> np.ndarray:
        """
        Paso: Smooth and binarize
        
        Aplica filtro Gaussiano y binariza la imagen.
        
        Args:
            data: Imagen de entrada
            sigma: Desviación estándar del filtro Gaussiano
            threshold: Umbral base para binarización
            
        Returns:
            Imagen binarizada (0 o 1)
        """
        smoothed = gaussian_filter(data, sigma)
        if np.max(smoothed) > 0:
            thresh = max(threshold, np.percentile(smoothed[smoothed > 0], 30))
        else:
            thresh = threshold
        return (smoothed > thresh).astype(np.float32)
    
    @staticmethod
    def dilate(binary_image: np.ndarray, distance: int) -> np.ndarray:
        """
        Paso: Dilate
        
        Expande (dilata) o contrae (erosiona) una imagen binaria.
        
        Args:
            binary_image: Imagen binaria
            distance: Distancia de dilatación (+ dilata, - erosiona, 0 closing)
            
        Returns:
            Imagen dilatada/erosionada
        """
        if distance == 0:
            return binary_closing(binary_image, disk(1)).astype(np.float32)
        elif distance < 0:
            closed = binary_closing(binary_image, disk(1))
            return binary_erosion(closed, disk(-distance)).astype(np.float32)
        else:
            return binary_dilation(binary_image, disk(distance)).astype(np.float32)
    
    @staticmethod
    def subtract(outer_layer: np.ndarray, inner_layer: np.ndarray, 
                 penalty: float = 100) -> np.ndarray:
        """
        Paso: Subtract (crear In-shell template)
        
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
    def create_in_shell(binary_mol: np.ndarray, 
                        insertion_distances: Tuple[int, int] = (-2, 0),
                        penalty: float = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pasos combinados: Smooth, Dilate, Subtract
        
        Crea el template "in-shell" completo.
        
        Args:
            binary_mol: Molécula binarizada
            insertion_distances: (inner_dist, outer_dist)
            penalty: Factor de penalización
            
        Returns:
            Tupla (template, outer_layer, inner_layer)
        """
        inner_dist, outer_dist = insertion_distances
        
        # Dilate para capa externa
        outer_layer = ImageProcessing.dilate(binary_mol, outer_dist)
        
        # Dilate para capa interna
        inner_layer = ImageProcessing.dilate(binary_mol, inner_dist)
        
        # Subtract
        template = ImageProcessing.subtract(outer_layer, inner_layer, penalty)
        
        return template, outer_layer, inner_layer
    
    @staticmethod
    def correlate(output_binary: np.ndarray, template: np.ndarray, 
                  box_size: int) -> np.ndarray:
        """
        Paso: Correlate, C-map
        
        Calcula el mapa de correlación entre el output y el template.
        
        Args:
            output_binary: Output actual binarizado
            template: Template in-shell
            box_size: Tamaño de la caja de la molécula (para enmascarar bordes)
            
        Returns:
            C-map (mapa de correlación con bordes enmascarados)
        """
        cmap = signal.correlate(output_binary, template, 'same', 'fft')
        
        # Enmascarar bordes
        half = box_size // 2
        cmap[:half+1, :] = 0
        cmap[:, :half+1] = 0
        cmap[-half-1:, :] = 0
        cmap[:, -half-1:] = 0
        
        return cmap
    
    @staticmethod
    def is_cmap_negative(cmap: np.ndarray) -> bool:
        """
        Paso: C-map is negative?
        
        Verifica si el máximo del C-map es negativo o cero (saturación).
        
        Args:
            cmap: Mapa de correlación
            
        Returns:
            True si saturación (no hay posición válida), False si hay espacio
        """
        return cmap.max() <= 0
    
    @staticmethod
    def find_maximum_position(cmap: np.ndarray) -> Tuple[int, int]:
        """
        Paso: Find maximum value entry of c-map
        
        Encuentra la posición del máximo en el C-map.
        
        Args:
            cmap: Mapa de correlación
            
        Returns:
            Coordenadas (x, y) del máximo
        """
        return np.unravel_index(cmap.argmax(), cmap.shape)
    
    @staticmethod
    def compute_frontier(binary_image: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Calcula la frontera (borde exterior) de una imagen binaria (GLOBAL).
        
        Args:
            binary_image: Imagen binaria
            radius: Radio de dilatación para la frontera
            
        Returns:
            Imagen de la frontera
        """
        if binary_image.max() == 0:
            return np.zeros_like(binary_image)
        
        dilated = binary_dilation(binary_image, disk(radius))
        frontier = dilated.astype(np.float32) - binary_image.astype(np.float32)
        
        return frontier
    
    @staticmethod
    def compute_frontier_local(binary_image: np.ndarray, 
                               position: Tuple[int, int],
                               box_size: int,
                               radius: int = 3,
                               margin: int = 5) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Calcula la frontera solo en la región afectada por una inserción (LOCAL).
        
        Más eficiente que recalcular toda la imagen.
        
        Args:
            binary_image: Imagen binaria completa
            position: Posición (x, y) donde se insertó la molécula
            box_size: Tamaño de la caja de la molécula
            radius: Radio de dilatación para la frontera
            margin: Margen extra alrededor de la región
            
        Returns:
            Tupla (frontera_local, (x_start, x_end, y_start, y_end))
        """
        x, y = position
        half = box_size // 2
        
        # Calcular región de interés con margen
        x_start = max(0, x - half - radius - margin)
        x_end = min(binary_image.shape[0], x + half + radius + margin)
        y_start = max(0, y - half - radius - margin)
        y_end = min(binary_image.shape[1], y + half + radius + margin)
        
        # Extraer región local
        local_region = binary_image[x_start:x_end, y_start:y_end]
        
        if local_region.max() == 0:
            return np.zeros_like(local_region), (x_start, x_end, y_start, y_end)
        
        # Calcular frontera solo en la región local
        dilated = binary_dilation(local_region, disk(radius))
        frontier_local = dilated.astype(np.float32) - local_region.astype(np.float32)
        
        return frontier_local, (x_start, x_end, y_start, y_end)
    
    @staticmethod
    def update_frontier_local(current_frontier: np.ndarray,
                              binary_image: np.ndarray,
                              position: Tuple[int, int],
                              box_size: int,
                              radius: int = 3,
                              margin: int = 5) -> np.ndarray:
        """
        Actualiza la frontera existente solo en la región afectada (LOCAL).
        
        En lugar de recalcular toda la frontera, solo actualiza la zona
        donde se insertó la nueva molécula.
        
        Args:
            current_frontier: Frontera actual (se modifica in-place)
            binary_image: Imagen binaria actualizada
            position: Posición donde se insertó la molécula
            box_size: Tamaño de la caja de la molécula
            radius: Radio de dilatación
            margin: Margen extra
            
        Returns:
            Frontera actualizada
        """
        # Calcular frontera local
        frontier_local, (x_start, x_end, y_start, y_end) = ImageProcessing.compute_frontier_local(
            binary_image, position, box_size, radius, margin
        )
        
        # Actualizar solo la región afectada
        updated_frontier = current_frontier.copy()
        updated_frontier[x_start:x_end, y_start:y_end] = frontier_local

        
        return updated_frontier

