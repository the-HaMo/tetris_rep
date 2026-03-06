# -*- coding: utf-8 -*-
"""
Parser 3D para archivos de configuracion de proteinas (.pns, .pms) y utilidades de salida.
"""

import os
from typing import Dict, List, Tuple, Optional

import mrcfile
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


class Parser3D:
    """
    Carga proteinas 3D y guarda salidas del Tetris 3D.
    """

    @staticmethod
    def parse_pns_file(filepath: str) -> Dict[str, str]:
        """
        Lee un archivo .pns/.pms y extrae sus parametros.
        """
        params: Dict[str, str] = {}
        with open(filepath, "r") as file_handle:
            for line in file_handle:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    params[key.strip()] = value.strip()
        return params

    @staticmethod
    def load_mrc_volume(filepath: str, swap_axes: bool = True) -> np.ndarray:
        """
        Carga un MRC como volumen 3D (numpy array).
        """
        with mrcfile.open(filepath, permissive=True) as mrc:
            data = mrc.data.copy()
        if swap_axes:
            data = np.swapaxes(data, 0, 2)
        return data

    @staticmethod
    def load_protein(filepath: str, base_dir: str) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Carga una proteina 3D desde .pns/.pms o .mrc.
        Retorna (volumen, parametros).
        """
        params: Dict[str, str] = {}
        if filepath.endswith(".pns") or filepath.endswith(".pms"):
            params = Parser3D.parse_pns_file(filepath)
            mrc_path = params.get("MMER_SVOL", "")
            if not mrc_path:
                raise ValueError("MMER_SVOL no encontrado en el archivo .pns/.pms")

            if mrc_path.startswith("/"):
                mrc_path = mrc_path[1:]

            full_mrc_path = os.path.join(base_dir, mrc_path)
            if not os.path.exists(full_mrc_path):
                raise FileNotFoundError(f"MRC no encontrado: {full_mrc_path}")

            volume = Parser3D.load_mrc_volume(full_mrc_path)
            return volume.astype(np.float32), params

        if filepath.endswith(".mrc"):
            volume = Parser3D.load_mrc_volume(filepath)
            return volume.astype(np.float32), params

        raise ValueError(f"Formato no soportado: {filepath}")

    @staticmethod
    def save_output_files(
        output_volume: np.ndarray,
        insertion_labels: np.ndarray,
        coordinates: List[Tuple[int, int, int]],
        molecule_types: List[str],
        filepath: str,
        voxel_size: float = 1.0,
        swap_axes: bool = True,
    ) -> None:
        """
        Guarda volumen, labels y coordenadas.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        out_volume = np.swapaxes(output_volume, 0, 2) if swap_axes else output_volume
        with mrcfile.new(filepath, overwrite=True) as mrc:
            mrc.set_data(out_volume.astype(np.float32))
            mrc.voxel_size = voxel_size

        labels_path = filepath.replace(".mrc", "_labels.mrc")
        out_labels = np.swapaxes(insertion_labels, 0, 2) if swap_axes else insertion_labels
        with mrcfile.new(labels_path, overwrite=True) as mrc:
            mrc.set_data(out_labels.astype(np.int16))
            mrc.voxel_size = voxel_size

        coords_path = filepath.replace(".mrc", "_coords.txt")
        with open(coords_path, "w") as file_handle:
            for idx, coord in enumerate(coordinates, start=1):
                label = molecule_types[idx - 1] if idx - 1 < len(molecule_types) else ""
                file_handle.write(f"{idx}: {coord} - {label}\n")

    @staticmethod
    def _numpy_to_vtk_image(volume: np.ndarray, voxel_size: float, swap_axes: bool = True) -> vtk.vtkImageData:
        data = np.swapaxes(volume, 0, 2) if swap_axes else volume
        data = np.ascontiguousarray(data.astype(np.float32))

        image = vtk.vtkImageData()
        image.SetDimensions(data.shape)
        image.SetSpacing(voxel_size, voxel_size, voxel_size)

        vtk_array = numpy_to_vtk(num_array=data.ravel(order="F"), deep=True)
        image.GetPointData().SetScalars(vtk_array)
        return image

    @staticmethod
    def _write_vtp_from_volume(
        volume: np.ndarray,
        output_path: str,
        voxel_size: float,
        iso_level: float,
        swap_axes: bool = True,
    ) -> None:
        image = Parser3D._numpy_to_vtk_image(volume, voxel_size, swap_axes)

        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(image)
        surface.SetValue(0, float(iso_level))
        surface.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(surface.GetOutput())
        if writer.Write() != 1:
            raise IOError(f"No se pudo escribir VTP: {output_path}")

    @staticmethod
    def save_vtp_files(
        output_volume: np.ndarray,
        insertion_labels: np.ndarray,
        output_dir: str,
        base_name: str,
        voxel_size: float = 1.0,
        iso_level: Optional[float] = None,
        sigma: float = 1.5,
        threshold: float = 50.0,
        swap_axes: bool = True,
    ) -> None:
        """
        Guarda superficies VTP de densidad y labels.
        """
        os.makedirs(output_dir, exist_ok=True)

        if iso_level is None:
            nonzero = output_volume[output_volume > 0]
            iso_level = float(np.percentile(nonzero, 70)) if nonzero.size else 0.0

        den_path = os.path.join(output_dir, f"{base_name}_den.vtp")
        if iso_level > 0:
            Parser3D._write_vtp_from_volume(
                output_volume, den_path, voxel_size, iso_level, swap_axes
            )

        skel_path = os.path.join(output_dir, f"{base_name}_skel.vtp")
        labels_binary = (insertion_labels > 0).astype(np.float32)
        if labels_binary.max() > 0:
            Parser3D._write_vtp_from_volume(
                labels_binary, skel_path, voxel_size, 0.5, swap_axes
            )
