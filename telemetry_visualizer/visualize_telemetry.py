#!/usr/bin/env python3
"""
Visualizador interactivo de telemetría con slider.

Uso:
  python visualize_telemetry.py "Artemis II Data.csv"
  python visualize_telemetry.py "Artemis II Data.csv" --index 100

Controles:
  - Slider: navegar por índices
  - Botones: Play/Pause, Prev, Next
  - Teclas: espacio=play/pause, izquierda/derecha=paso
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Forzar backend ANTES de importar pyplot
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle


# ===== CONFIGURACIÓN GLOBAL =====
PLAYBACK_SPEED = 200  # milliseconds entre frames durante reproducción
INFO_BOX_X = 1.1    # posición x del cartel informativo (0-1)
INFO_BOX_Y = 0.95    # posición y del cartel informativo (0-1)
VELOCITY_MULTIPLIER = 1.0  # multiplicador para el tamaño del vector de velocidad
Y_OFFSET = 0  # desplazamiento vertical del gráfico
GRAPH_TOP = 0.95  # posición del borde superior del gráfico (0-1)
GRAPH_BOTTOM = 0.18  # posición del borde inferior del gráfico (0-1)
GRAPH_LEFT = 0.1  # posición del borde izquierdo del gráfico (0-1)
GRAPH_RIGHT = 0.95  # posición del borde derecho del gráfico (0-1)
BODY_RADIUS_KM = 6371  # radio del cuerpo celeste en km (1737=Luna, 6371=Tierra)
BODY_RADIUS_MILES = BODY_RADIUS_KM / 1.609  # radio en millas
# ================================


def parse_args():
    p = argparse.ArgumentParser(description='Visualizador interactivo de telemetría')
    p.add_argument('file', nargs='?', default='Artemis II Data.csv', 
                   help='CSV file (sep=";", decimal=",")')
    p.add_argument('--index', type=int, default=0, help='Índice inicial')
    return p.parse_args()


def load_csv(path):
    """Load Artemis II CSV: sep=';', decimal=',', columnas: time,x,y,z,vx,vy,vz"""
    df = pd.read_csv(path, sep=';', decimal=',', header=None,
                     names=['time','x','y','z','vx','vy','vz'], dtype={'time': str})
    df['time'] = pd.to_datetime(df['time'].str.replace(',', '.'), errors='coerce')
    for c in ['x','y','z','vx','vy','vz']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['speed'] = np.hypot(df['vx'], df['vy'])
    return df


def main():
    args = parse_args()
    
    df = load_csv(args.file)
    if df.empty:
        print('Error: archivo vacío o lectura fallida', file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Maximizar ventana
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    # Ajustar márgenes del gráfico
    plt.subplots_adjust(bottom=GRAPH_BOTTOM, top=GRAPH_TOP, left=GRAPH_LEFT, right=GRAPH_RIGHT)
    
    # Graficar trayectoria completa
    ax.plot(df['x'], df['y'], lw=0.6, color='0.8', label='trayectoria')
    point, = ax.plot([], [], 'ro', ms=6)
    
    # Agregar círculo en origen (0,0)
    circle = Circle((0, 0), BODY_RADIUS_MILES, fill=False, edgecolor='blue', linewidth=1, linestyle='-')
    ax.add_patch(circle)

    # Calcular escala para el vector de velocidad
    dx = float(df['x'].max() - df['x'].min())
    dy = float(df['y'].max() - df['y'].min())
    span = max(dx, dy) if max(dx, dy) > 0 else 1.0
    vmax = float(df['speed'].replace(0, np.nan).max()) if df['speed'].notna().any() else 1.0
    vel_scale = 0.15 * span / vmax * VELOCITY_MULTIPLIER if np.isfinite(vmax) and vmax > 0 else 1.0

    # Hacer que el gráfico sea cuadrado
    center_x = (df['x'].min() + df['x'].max()) / 2
    center_y = (df['y'].min() + df['y'].max()) / 2 + Y_OFFSET
    half_span = span / 2
    ax.set_xlim(center_x - half_span, center_x + half_span)
    ax.set_ylim(center_y - half_span, center_y + half_span)

    # Índice inicial válido
    i0 = max(0, min(len(df)-1, args.index))
    quiv = ax.quiver(df.at[i0,'x'], df.at[i0,'y'],
                     df.at[i0,'vx'] * vel_scale, df.at[i0,'vy'] * vel_scale,
                     angles='xy', scale_units='xy', scale=1, color='r')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Telemetría: posición (punto rojo) y velocidad (flecha roja)')
    
    # Rotar etiquetas del eje x a 90 grados
    ax.tick_params(axis='x', rotation=90)

    # Caja de información
    info = ax.text(INFO_BOX_X, INFO_BOX_Y, '', transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Slider
    ax_slider = plt.axes([0.56, 0.01, 0.40, 0.04])
    slider = Slider(ax_slider, 'Índice', 0, len(df)-1, valinit=i0, 
                    valfmt='%0.0f', valstep=1)

    def update(val):
        idx = int(slider.val)
        x = df.at[idx, 'x']
        y = df.at[idx, 'y']
        vx = df.at[idx, 'vx']
        vy = df.at[idx, 'vy']
        sp = df.at[idx, 'speed']
        
        point.set_data([x], [y])
        quiv.set_offsets(np.array([[x, y]]))
        quiv.set_UVC(np.array([vx * vel_scale]), np.array([vy * vel_scale]))
        
        t = df.at[idx, 'time']
        info.set_text(f'idx={idx}\n{t}\n(x,y)=({x:.0f}, {y:.0f})\n(vx,vy)=({vx:.3f}, {vy:.3f})\n|v|={sp:.3f}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(i0)

    # Botones
    ax_play = plt.axes([0.12, 0.01, 0.12, 0.04])
    btn_play = Button(ax_play, 'Play')
    
    ax_prev = plt.axes([0.26, 0.01, 0.12, 0.04])
    btn_prev = Button(ax_prev, 'Prev')
    
    ax_next = plt.axes([0.40, 0.01, 0.12, 0.04])
    btn_next = Button(ax_next, 'Next')

    playing = {'val': False}
    timer = fig.canvas.new_timer(interval=PLAYBACK_SPEED)

    def _tick():
        idx = int(slider.val)
        if idx >= len(df) - 1:
            timer.stop()
            playing['val'] = False
            btn_play.label.set_text('Play')
            return
        slider.set_val(min(len(df) - 1, idx + 1))

    timer.add_callback(_tick)

    def on_play(event):
        if not playing['val']:
            playing['val'] = True
            btn_play.label.set_text('Pause')
            timer.start()
        else:
            playing['val'] = False
            btn_play.label.set_text('Play')
            timer.stop()

    def on_prev(event):
        idx = int(slider.val)
        slider.set_val(max(0, idx - 1))

    def on_next(event):
        idx = int(slider.val)
        slider.set_val(min(len(df) - 1, idx + 1))

    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_key(event):
        if event.key in (' ', 'k'):
            on_play(None)
        elif event.key in ('left', 'a'):
            on_prev(None)
        elif event.key in ('right', 'd'):
            on_next(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == '__main__':
    main()
