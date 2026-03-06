# -*- coding: utf-8 -*-
"""
Parser para archivos de configuración de proteínas (.pns, .pms)
"""

import os
import numpy as np
import mrcfile


class ProteinParser:
    """
    Clase para parsear archivos de configuración de proteínas y cargar MRCs.
    """
    
    @staticmethod
    def parse_pns_file(filepath: str) -> dict:
        """
        Lee un archivo .pns/.pms y extrae sus parámetros.
        Retorna un diccionario con las claves del archivo.
        """
        params = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    params[key.strip()] = value.strip()
        return params
    
    @staticmethod
    def load_mrc_as_2d(filepath: str) -> np.ndarray:
        """Carga MRC y proyecta a 2D."""
        with mrcfile.open(filepath) as mrc:
            data = mrc.data.copy()
        proj = np.max(data, axis=0)
        if proj.max() > proj.min():
            proj = (proj - proj.min()) / (proj.max() - proj.min()) * 1000
        return proj.astype(np.float32)
    
    @staticmethod
    def load_protein(filepath: str, base_dir: str) -> np.ndarray:
        """
        Carga una proteína desde un archivo .pns, .pms o .mrc.
        - Si es .pns/.pms: lee la ruta del MRC desde MMER_SVOL
        - Si es .mrc: lo carga directamente
        """
        if filepath.endswith('.pns') or filepath.endswith('.pms'):
            # Leer archivo de configuración
            params = ProteinParser.parse_pns_file(filepath)
            mrc_path = params.get('MMER_SVOL', '')
            
            # La ruta en el .pns puede ser relativa a data/
            if mrc_path.startswith('/'):
                mrc_path = mrc_path[1:]  # Quitar / inicial
            
            full_mrc_path = os.path.join(base_dir, mrc_path)
            
            if not os.path.exists(full_mrc_path):
                raise FileNotFoundError(f"MRC no encontrado: {full_mrc_path}")
            
            return ProteinParser.load_mrc_as_2d(full_mrc_path)
        else:
            # Archivo MRC directo
            return ProteinParser.load_mrc_as_2d(filepath)