import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Backend interactivo compatible

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle

# Cargar datos del CSV
if len(sys.argv) < 2:
    print("Uso: python visualize_orbit_orion.py <csv_orion> [csv_luna]")
    print("Ejemplo: python visualize_orbit_orion.py trayectoria_euler_artemis.csv ../luna/trayectoria_euler_luna.csv")
    sys.exit(1)

archivo_orion = sys.argv[1]
archivo_luna = sys.argv[2] if len(sys.argv) > 2 else '../luna/trayectoria_euler_luna.csv'

print(f"Cargando Orion desde: {archivo_orion}")
print(f"Cargando Luna desde: {archivo_luna}")

try:
    datos_orion = np.loadtxt(archivo_orion, delimiter=',', skiprows=1)
    print(f"✓ Orion cargado: {len(datos_orion)} puntos")
except Exception as e:
    print(f"✗ Error al cargar Orion: {e}")
    sys.exit(1)

try:
    datos_luna = np.loadtxt(archivo_luna, delimiter=',', skiprows=1)
    print(f"✓ Luna cargada: {len(datos_luna)} puntos")
except Exception as e:
    print(f"✗ Error al cargar Luna: {e}")
    sys.exit(1)

tiempos = datos_orion[:, 0]
x = datos_orion[:, 1]
y = datos_orion[:, 2]
vx = datos_orion[:, 3]
vy = datos_orion[:, 4]

# Interpolar Luna al mismo tiempo que Orion
tiempos_luna = datos_luna[:, 0]
x_luna_raw = datos_luna[:, 1]
y_luna_raw = datos_luna[:, 2]

# Parámetros
R_EARTH = 6.371e6  # Radio de la Tierra en metros
SCALE_VEL = 1e4    # Escala para los vectores de velocidad

# Crear figura principal
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(left=0.12, bottom=0.25, right=0.95, top=0.95)

# Graficar la trayectoria completa
ax.plot(x, y, 'b-', alpha=0.3, linewidth=1, label='Trayectoria orbital')

# Tierra como círculo
earth = Circle((0, 0), R_EARTH, color='blue', alpha=0.3, label='Tierra')
ax.add_patch(earth)

# Punto actual
point, = ax.plot([], [], 'ro', markersize=10, label='Posición actual', zorder=5)

# Vector de velocidad (quiver)
quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1e-4, 
                   color='red', width=0.006, label='Vector velocidad', zorder=4)

# Texto informativo
info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8), family='monospace')

# Configurar ejes
ax.set_xlabel('Posición X [m]', fontsize=12)
ax.set_ylabel('Posición Y [m]', fontsize=12)
ax.set_title('Órbita Lunar - Método de Euler', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=10)

# Establecer límites
margen = 0.1e8
ax.set_xlim([x.min() - margen, x.max() + margen])
ax.set_ylim([y.min() - margen, y.max() + margen])

# Crear slider
ax_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
slider = Slider(ax_slider, 'Punto', 0, len(tiempos) - 1, valinit=0, 
                valstep=1, color='orange', valfmt='%0.0f')

def update(val):
    """Actualiza la visualización basada en el slider"""
    idx = int(slider.val)
    
    # Actualizar punto
    point.set_data([x[idx]], [y[idx]])
    
    # Actualizar vector de velocidad
    u = vx[idx] * SCALE_VEL
    v = vy[idx] * SCALE_VEL
    quiver.set_offsets([[x[idx], y[idx]]])
    quiver.set_UVC([[u]], [[v]])
    
    # Información de tiempo
    tiempo_horas = tiempos[idx] / 3600
    tiempo_dias = tiempos[idx] / (24 * 3600)
    magnitud_v = np.sqrt(vx[idx]**2 + vy[idx]**2)
    
    info_text.set_text(
        f'Índice: {idx}/{len(tiempos)-1}\n'
        f'Tiempo: {tiempo_horas:.1f} h ({tiempo_dias:.3f} días)\n'
        f'Pos: ({x[idx]:.2e}, {y[idx]:.2e}) m\n'
        f'Vel: ({vx[idx]:.1f}, {vy[idx]:.1f}) m/s\n'
        f'|V|: {magnitud_v:.1f} m/s'
    )
    
    fig.canvas.draw_idle()

slider.on_changed(update)

# Botones de control
ax_play = plt.axes([0.15, 0.08, 0.06, 0.04])
btn_play = Button(ax_play, 'Play')

ax_stop = plt.axes([0.22, 0.08, 0.06, 0.04])
btn_stop = Button(ax_stop, 'Stop')

ax_reset = plt.axes([0.29, 0.08, 0.06, 0.04])
btn_reset = Button(ax_reset, 'Reset')

ax_prev = plt.axes([0.36, 0.08, 0.06, 0.04])
btn_prev = Button(ax_prev, '◀ Prev')

ax_next = plt.axes([0.43, 0.08, 0.06, 0.04])
btn_next = Button(ax_next, 'Next ▶')

# Variables de animación
animation_id = None
is_playing = False

def play(event):
    global animation_id, is_playing
    if is_playing:
        return
    is_playing = True
    
    def animate():
        global animation_id, is_playing
        if is_playing:
            current_idx = int(slider.val)
            if current_idx < len(tiempos) - 1:
                slider.set_val(current_idx + 1)
                animation_id = fig.canvas.new_timer()
                animation_id.interval = 50
                animation_id.single_shot = True
                animation_id.callback(animate)
                animation_id.start()
            else:
                is_playing = False
    
    animate()

def stop(event):
    global is_playing
    is_playing = False

def reset(event):
    global is_playing
    is_playing = False
    slider.set_val(0)

def prev_step(event):
    global is_playing
    is_playing = False
    current_idx = int(slider.val)
    if current_idx > 0:
        slider.set_val(current_idx - 1)

def next_step(event):
    global is_playing
    is_playing = False
    current_idx = int(slider.val)
    if current_idx < len(tiempos) - 1:
        slider.set_val(current_idx + 1)

btn_play.on_clicked(play)
btn_stop.on_clicked(stop)
btn_reset.on_clicked(reset)
btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

# Inicializar
update(0)

plt.show()
