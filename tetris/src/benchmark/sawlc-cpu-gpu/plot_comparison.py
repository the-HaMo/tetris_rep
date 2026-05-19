import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
from matplotlib.lines import Line2D

# Base de la membrana detectada en tus logs para el Tomograma 3
MEMBRANE_LEVEL = 11.5450 

def parse_detailed_log(file_path, algo_type):
    """Extrae métricas finales y detalle de monómeros por tipo con Regex robusta."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except: return None
    
    data = {'num_types': 0, 'total_occ': 0, 'time_min': 0, 'total_monomers': 0,
            'monomers_detail': {}, 'prot_name': ''}
    
    # 1. EXTRACCIÓN DE TIEMPO (Regex flexible)
    if algo_type in ['Tetris', 'Tetris GPU']:
        time_m = re.search(r"DONE:.*?in\s+(?:(\d+)m\s+)?([\d.]+)\s*s", content)
        pattern = r"After type \d+ \((.*?)\):.*?num_monomers=(\d+)"
    else:
        time_m = re.search(r"TOTAL PROCESSING TIME:\s+(?:(\d+)min\s+)?([\d.]+)\s*s", content)
        pattern = r"After type \d+ \((.*?)\):.*?monomers=(\d+)"

    if time_m:
        mins = float(time_m.group(1)) if time_m.group(1) else 0.0
        secs = float(time_m.group(2)) if time_m.group(2) else 0.0
        data['time_min'] = mins + (secs / 60.0)
        if data['time_min'] <= 0: data['time_min'] = 0.0001 
    else:
        data['time_min'] = 1.0 

    # 2. Ocupancia FINAL
    occ_values = re.findall(r"total_occ\s*[:=]\s*([\d.]+)%", content)
    if occ_values:
        data['total_occ'] = float(occ_values[-1])
    
    # 3. Detalle de Monómeros
    lines = re.findall(pattern, content)
    for name, count in lines:
        short_name = name.split('_')[0]
        c = int(count)
        data['monomers_detail'][short_name] = c
        data['total_monomers'] += c
        data['prot_name'] = short_name  # última proteína añadida = la nueva en esta sim
    
    data['num_types'] = len(lines)
    return data

def load_data(directory, label):
    if not os.path.exists(directory): 
        print(f"⚠️  Carpeta no encontrada: {directory}")
        return []
    files = sorted([f for f in os.listdir(directory) if 'tomo3_den' in f and f.endswith('.txt')],
                   key=lambda x: int(re.search(r'den(\d+)', x).group(1)))
    return [parse_detailed_log(os.path.join(directory, f), label) for f in files]

# --- PROCESAMIENTO ---
tetris_results = load_data('logs_tetris', 'Tetris')
tetris_gpu_results = load_data('logs_tetris_gpu', 'Tetris GPU')
sawlc_results = load_data('logs_sawlc', 'SAWLC')

df_t = pd.DataFrame(tetris_results)
df_tgpu = pd.DataFrame(tetris_gpu_results)
df_s = pd.DataFrame(sawlc_results)

# Etiquetas del eje X: nombres de proteínas acumulativos (igual que la imagen de referencia)
# Usamos los nombres de la lista más larga disponible
def get_prot_labels(results):
    """Devuelve la lista de nombres de proteína añadidos en cada simulación."""
    labels = []
    for sim in results:
        keys = list(sim['monomers_detail'].keys())
        labels.append(keys[-1] if keys else str(sim['num_types']))
    return labels

# Tomamos las etiquetas de Tetris CPU (o la que esté disponible)
ref_results = tetris_results or tetris_gpu_results or sawlc_results
x_labels = get_prot_labels(ref_results)
x_pos = list(range(len(x_labels)))

def get_x(results, ref_labels):
    """Mapea cada simulación a su posición en el eje X por nombre de proteína."""
    positions, labels = [], get_prot_labels(results)
    for lbl in labels:
        if lbl in ref_labels:
            positions.append(ref_labels.index(lbl))
        else:
            positions.append(len(ref_labels) - 1)
    return positions

x_t    = get_x(tetris_results, x_labels)
x_tgpu = get_x(tetris_gpu_results, x_labels)
x_s    = get_x(sawlc_results, x_labels)

# Mapeo de colores único para proteínas
all_prots = sorted(list(set(
    [p for s in tetris_results for p in s['monomers_detail'].keys()] +
    [p for s in tetris_gpu_results for p in s['monomers_detail'].keys()] +
    [p for s in sawlc_results for p in s['monomers_detail'].keys()]
)))
color_map = dict(zip(all_prots, plt.cm.tab20(np.linspace(0, 1, len(all_prots)))))

# Helper para configurar eje X con nombres de proteínas
def set_prot_xaxis(ax, x_labels):
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Tipos de Proteína', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, len(x_labels) - 0.5)

# --- GENERACIÓN DE GRÁFICAS ---
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('TETRIS (CPU/GPU) vs SAWLC\n',
             fontsize=18, fontweight='bold')
plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.90)

for ax in axs.flat:
    ax.grid(True, alpha=0.3)

# 1. SATURACIÓN TOTAL
ax0 = axs[0, 0]
ax0.axhline(y=MEMBRANE_LEVEL, color='black', linestyle=':', alpha=0.5, label='Membrana (11.5%)')
if not df_t.empty:
    ax0.plot(x_t, df_t['total_occ'], 'o-', color='blue', lw=2, label='Tetris CPU')
if not df_tgpu.empty:
    ax0.plot(x_tgpu, df_tgpu['total_occ'], 'D-', color='purple', lw=2, label='Tetris GPU')
if not df_s.empty:
    ax0.plot(x_s, df_s['total_occ'], 's--', color='orange', lw=2, label='SAWLC')
ax0.set_title('Saturación Alcanzada', fontsize=14, fontweight='bold')
ax0.set_ylabel('Ocupancia Total (%)')
ax0.set_ylim(0, 65)
ax0.legend(fontsize=10)
set_prot_xaxis(ax0, x_labels)

# 2. TIEMPO DE EJECUCIÓN
ax1 = axs[0, 1]
if not df_t.empty:
    ax1.plot(x_t, df_t['time_min'], 'o-', color='blue', lw=2, label='Tetris CPU')
if not df_tgpu.empty:
    ax1.plot(x_tgpu, df_tgpu['time_min'], 'D-', color='purple', lw=2, label='Tetris GPU')
if not df_s.empty:
    ax1.plot(x_s, df_s['time_min'], 's--', color='orange', lw=2, label='SAWLC')
ax1.set_title('Tiempo de Ejecución', fontsize=14, fontweight='bold')
ax1.set_ylabel('Minutos')
ax1.legend(fontsize=10)
set_prot_xaxis(ax1, x_labels)

# 3. POBLACIÓN DE PROTEÍNAS (BARRAS APILADAS)
ax_bar = axs[1, 0]
width = 0.25
configs = [
    (tetris_results,     x_t,    -width, 'white', 'Tetris CPU'),
    (tetris_gpu_results, x_tgpu,  0,     'cyan',  'Tetris GPU'),
    (sawlc_results,      x_s,     width, 'black', 'SAWLC'),
]

for sim_list, x_positions, offset, edge_c, lab in configs:
    for i, sim in enumerate(sim_list):
        bottom = 0
        x = x_positions[i] + offset
        for p in all_prots:
            val = sim['monomers_detail'].get(p, 0)
            if val > 0:
                ax_bar.bar(x, val, width, bottom=bottom,
                           color=color_map[p], edgecolor=edge_c, lw=0.7)
                bottom += val

ax_bar.set_title('Población de Monómeros', fontsize=14, fontweight='bold')
ax_bar.set_ylabel('Cantidad')
set_prot_xaxis(ax_bar, x_labels)

# Leyenda 1: colores de proteínas
leg_prots = ax_bar.legend(
    handles=[Line2D([0], [0], color=color_map[p], lw=6, label=p) for p in all_prots],
    title="Tipos de Proteína", loc='upper left', fontsize=7, ncol=2
)

ax_bar.add_artist(leg_prots)
ax_bar.legend(title="Algoritmos", loc='upper right', fontsize=8)

# 4. RENDIMIENTO
ax3 = axs[1, 1]
if not df_t.empty:
    ax3.plot(x_t, df_t['total_monomers'] / df_t['time_min'], 'o-', color='blue', lw=2, label='Tetris CPU')
if not df_tgpu.empty:
    ax3.plot(x_tgpu, df_tgpu['total_monomers'] / df_tgpu['time_min'], 'D-', color='purple', lw=2, label='Tetris GPU')
if not df_s.empty:
    ax3.plot(x_s, df_s['total_monomers'] / df_s['time_min'], 's--', color='orange', lw=2, label='SAWLC')
ax3.set_title('Rendimiento (Proteínas/min)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Proteínas / min')
ax3.legend(fontsize=10)
set_prot_xaxis(ax3, x_labels)

# --- GUARDADO FINAL ---
output_img = "resultados_comparativa_completa.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"\n✅ Proceso terminado. Imagen guardada como: {output_img}")
