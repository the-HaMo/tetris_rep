import imageio.v2 as imageio
import glob
import os
import sys

# Tomamos la ruta de la carpeta desde el comando
folder = sys.argv[1] if len(sys.argv) > 1 else '.'

# Buscamos los archivos con el nombre exacto que puso mrc2tif
search_path = os.path.join(folder, "frame.*.png")
archivos = sorted(glob.glob(search_path))

if not archivos:
    print(f"Error: No se encontraron archivos en: {os.path.abspath(search_path)}")
else:
    print(f"Éxito: Procesando {len(archivos)} imágenes...")
    
    # Para que dure más, subimos la duración (ej: 0.1 o 0.2)
    with imageio.get_writer('tetris_final.gif', mode='I', duration=0.1) as writer:
        for f in archivos:
            writer.append_data(imageio.imread(f))
    
    print("¡GIF guardado como tetris_final.gif en la carpeta src!")