import numpy as np
from scipy import signal

class Tetris3D:
    def __init__(self, dimensions=(500, 500, 250), sigma=1.5, threshold=100):
        self.dimensions = np.array(dimensions)
        self.sigma = sigma
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.output_volume = np.zeros(self.dimensions, dtype=np.float32)
        self.all_coordinates = []

    def get_occupancy(self) -> float:
        occupied = np.count_nonzero(self.output_volume > self.threshold)
        return occupied / self.output_volume.size

    def place_molecule_3d(self, position, molecule, molecule_id):
        z, y, x = position
        m_shape = np.array(molecule.shape)
        half = m_shape // 2
        # Coordenadas exactas garantizadas por el motor de inserción
        z_s, z_e = z-half[0], z-half[0]+m_shape[0]
        y_s, y_e = y-half[1], y-half[1]+m_shape[1]
        x_s, x_e = x-half[2], x-half[2]+m_shape[2]
        self.output_volume[z_s:z_e, y_s:y_e, x_s:x_e] += molecule

    def insert_molecule_3d(self, molecule_template, molecule_rotated, mol_name, allowed_mask, target_coord, box_size):
        step = len(self.all_coordinates) + 1
        z_t, y_t, x_t = target_coord
        pad = int(box_size * 1.5) 
        z_s, z_e = max(0, z_t-pad), min(self.dimensions[0], z_t+pad)
        y_s, y_e = max(0, y_t-pad), min(self.dimensions[1], y_t+pad)
        x_s, x_e = max(0, x_t-pad), min(self.dimensions[2], x_t+pad)

        local_vol = self.output_volume[z_s:z_e, y_s:y_e, x_s:x_e]
        local_bin = (local_vol > self.threshold).astype(np.float32)

        # Semilla inicial: chequeo estricto de integridad
        if local_bin.max() == 0:
            h = np.array(molecule_rotated.shape) // 2
            dz_s, dz_e = z_t-h[0], z_t-h[0]+molecule_rotated.shape[0]
            dy_s, dy_e = y_t-h[1], y_t-h[1]+molecule_rotated.shape[1]
            dx_s, dx_e = x_t-h[2], x_t-h[2]+molecule_rotated.shape[2]
            
            if (dz_s >= 0 and dz_e <= self.dimensions[0] and dy_s >= 0 and dy_e <= self.dimensions[1] and dx_s >= 0 and dx_e <= self.dimensions[2]):
                if not np.any(self.output_volume[dz_s:dz_e, dy_s:dy_e, dx_s:dx_e] > self.threshold):
                    self.place_molecule_3d((z_t, y_t, x_t), molecule_rotated, step)
                    self.all_coordinates.append((z_t, y_t, x_t))
                    return 'inserted'
            return 'saturated'

        cmap = signal.correlate(local_bin, molecule_template, mode='same', method='fft')
        cmap = np.where(allowed_mask[z_s:z_e, y_s:y_e, x_s:x_e], cmap, -1e9) 

        temp_cmap = cmap.copy()
        for _ in range(50):
            if temp_cmap.max() <= -1e8: return 'saturated'
            idx = np.unravel_index(temp_cmap.argmax(), temp_cmap.shape)
            z, y, x = idx[0] + z_s, idx[1] + y_s, idx[2] + x_s
            h = np.array(molecule_rotated.shape) // 2
            dz_s, dz_e = z-h[0], z-h[0]+molecule_rotated.shape[0]
            dy_s, dy_e = y-h[1], y-h[1]+molecule_rotated.shape[1]
            dx_s, dx_e = x-h[2], x-h[2]+molecule_rotated.shape[2]

            # GARANTÍA: Si se sale del cuadro, descarta. No corta proteínas.
            if (dz_s < 0 or dz_e > self.dimensions[0] or dy_s < 0 or dy_e > self.dimensions[1] or dx_s < 0 or dx_e > self.dimensions[2]):
                temp_cmap[idx] = -1e9; continue
            
            if np.any((self.output_volume[dz_s:dz_e, dy_s:dy_e, dx_s:dx_e] > self.threshold) & (molecule_rotated > self.threshold)):
                temp_cmap[idx] = -1e9; continue
            
            self.place_molecule_3d((z, y, x), molecule_rotated, step)
            self.all_coordinates.append((z, y, x))
            print(f"    -> Paso {step}: {mol_name} en {(z,y,x)} | Occ: {self.get_occupancy()*100:.4f}%")
            return 'inserted'
        return 'saturated'