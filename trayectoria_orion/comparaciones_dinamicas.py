#!/usr/bin/env python3
"""
Comparador de trayectorias orbitales en PyQt5 con Slider y Barra de Zoom Interactiva.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Configurar backend de Matplotlib para PyQt5
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# CAMBIO CLÍTICO: Importar la barra de herramientas de navegación para Qt5
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Importar componentes de PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSlider, QMessageBox)
from PyQt5.QtCore import Qt

# ── Constantes físicas ───────────────────────────────────────────────────────
CONST_G = 6.674e-11
M_T     = 5.972e24
M_L     = 7.342e22
EARTH_RADIUS_KM = 6371
MOON_RADIUS_KM = 1737

PALETA = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

# ────────────────────────────────────────────────────────────────────────────
# FÍSICA E INTEGRADORES (Igual al original)
# ────────────────────────────────────────────────────────────────────────────
def gravitational_acceleration(x, y, pos_luna):
    r2_t = x**2 + y**2
    r_t  = r2_t**0.5
    if r_t == 0: ax_t = ay_t = 0.0
    else:
        a_t  = CONST_G * M_T / r2_t
        ax_t = -a_t * (x / r_t)
        ay_t = -a_t * (y / r_t)

    dx   = x - pos_luna[0]
    dy   = y - pos_luna[1]
    r2_l = dx**2 + dy**2
    r_l  = r2_l**0.5
    if r_l == 0: ax_l = ay_l = 0.0
    else:
        a_l  = CONST_G * M_L / r2_l
        ax_l = -a_l * (dx / r_l)
        ay_l = -a_l * (dy / r_l)
    return (ax_t + ax_l, ay_t + ay_l)

def derivadas(estado, pos_luna):
    x, y, vx, vy = estado
    ax, ay = gravitational_acceleration(x, y, pos_luna)
    return np.array([vx, vy, ax, ay])

def euler_step(estado, dt, pos_luna): return estado + dt * derivadas(estado, pos_luna)
def rk2_step(estado, dt, pos_luna):
    k1 = derivadas(estado, pos_luna)
    k2 = derivadas(estado + dt * k1, pos_luna)
    return estado + dt * 0.5 * (k1 + k2)
def rk4_step(estado, dt, pos_luna):
    k1 = derivadas(estado,              pos_luna)
    k2 = derivadas(estado + dt/2 * k1, pos_luna)
    k3 = derivadas(estado + dt/2 * k2, pos_luna)
    k4 = derivadas(estado + dt   * k3, pos_luna)
    return estado + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

INTEGRADORES = {'euler': euler_step, 'rk2': rk2_step, 'rk4': rk4_step}

# ────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS Y SIMULACIÓN
# ────────────────────────────────────────────────────────────────────────────
def cargar_datos_luna(ruta: str) -> np.ndarray:
    df = pd.read_csv(ruta, low_memory=False)
    col_map = {}
    for col in df.columns:
        cu = col.strip().upper()
        if cu in ('X_LUNA', 'X'):  col_map[col] = 'x'
        elif cu in ('Y_LUNA', 'Y'): col_map[col] = 'y'
    df = df.rename(columns=col_map)
    for c in ('x', 'y'): df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['x', 'y'])
    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3
    return df[['x', 'y']].to_numpy(), df[['x', 'y']].reset_index(drop=True)

def simular(nombre: str, dt: float, duracion_dias: float,
            estado_inicial: np.ndarray, datos_luna: np.ndarray) -> pd.DataFrame:
    integrador = INTEGRADORES[nombre]
    pasos      = int(duracion_dias * 86400.0 / dt)
    n_luna     = len(datos_luna)
    estado = estado_inicial.copy()
    t      = 0.0
    registros = np.empty((pasos + 1, 5))
    registros[0] = [t, *estado]

    for i in range(pasos):
        pos_luna = datos_luna[-1] if i >= n_luna else datos_luna[i]
        estado = integrador(estado, dt, pos_luna)
        t     += dt
        registros[i + 1] = [t, *estado]
    return pd.DataFrame(registros, columns=['time_s', 'x', 'y', 'vx', 'vy'])

def cargar_csv_trayectoria(ruta: str) -> pd.DataFrame:
    with open(ruta) as f: lineas = f.readlines()
    sep = ';' if any(';' in l for l in lineas[:5]) else ','
    if sep == ';':
        df = pd.read_csv(ruta, sep=';', decimal=',', header=None, names=['time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    else:
        header_idx = None
        for i, linea in enumerate(lineas):
            partes = [p.strip().upper() for p in linea.split(',')]
            if 'X' in partes and 'Y' in partes:
                header_idx = i
                break
        if header_idx is None: sys.exit(1)
        df = pd.read_csv(ruta, sep=',', header=header_idx, low_memory=False)
        col_map = {}
        for col in df.columns:
            cu = col.strip().upper()
            if cu in ('JDTDB', 'TIME_S', 'TIME'): col_map[col] = 'time_s'
            elif cu == 'X':  col_map[col] = 'x'
            elif cu == 'Y':  col_map[col] = 'y'
        df = df.rename(columns=col_map)
    for c in ('x', 'y'): df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['x', 'y'])
    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3
    return df

def resolver_fuentes(metodos, dt, duracion, estado_inicial, exportar_csv=False):
    resultados = []
    for m in metodos:
        if m.startswith('csv:'):
            ruta = m[4:]
            if os.path.exists(ruta):
                df = cargar_csv_trayectoria(ruta)
                resultados.append((os.path.splitext(os.path.basename(ruta))[0], df, None))
        elif m in INTEGRADORES:
            ruta_luna = f"resultado_{m.lower()}.csv"
            if not os.path.exists(ruta_luna): continue
            datos_luna_arr, df_luna = cargar_datos_luna(ruta_luna)
            df = simular(m, dt, duracion, estado_inicial, datos_luna_arr)
            resultados.append((m.upper(), df, df_luna))
    return resultados


# ────────────────────────────────────────────────────────────────────────────
# INTERFAZ GRÁFICA EN PYQT5 CON ZOOM
# ────────────────────────────────────────────────────────────────────────────

class VentanaOrbital(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fuentes = []
        
        self.setWindowTitle("Simulador Orbital Interactiva (Zoom Habilitado)")
        self.resize(950, 850)

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # --- Panel Superior de Controles ---
        layout_controles = QHBoxLayout()
        layout_controles.addWidget(QLabel("extra_v[0] (vx, m/s):"))
        self.input_v0 = QLineEdit("-460")
        self.input_v0.setFixedWidth(80)
        layout_controles.addWidget(self.input_v0)

        layout_controles.addWidget(QLabel("extra_v[1] (vy, m/s):"))
        self.input_v1 = QLineEdit("-150")
        self.input_v1.setFixedWidth(80)
        layout_controles.addWidget(self.input_v1)

        self.btn_recalcular = QPushButton("Recalcular")
        self.btn_recalcular.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")
        self.btn_recalcular.clicked.connect(self.ejecutar_recalculo)
        layout_controles.addWidget(self.btn_recalcular)
        layout_controles.addStretch()
        layout_principal.addLayout(layout_controles)

        # --- Lienzo del Gráfico ---
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        
        # NUEVO: Añadir la barra de herramientas de zoom nativa en el layout
        # Recibe el canvas actual y la ventana contenedora (self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_principal.addWidget(self.toolbar)
        layout_principal.addWidget(self.canvas)

        # --- Panel Inferior del Slider ---
        layout_slider = QHBoxLayout()
        self.label_tiempo = QLabel("Paso: 0 / 0")
        self.label_tiempo.setFixedWidth(150)
        layout_slider.addWidget(self.label_tiempo)

        self.slider_tiempo = QSlider(Qt.Horizontal)
        self.slider_tiempo.setMinimum(0)
        self.slider_tiempo.setMaximum(100)
        self.slider_tiempo.setEnabled(False)
        self.slider_tiempo.valueChanged.connect(self.actualizar_marcador_tiempo)
        layout_slider.addWidget(self.slider_tiempo)
        layout_principal.addLayout(layout_slider)

        self.marcadores_moviles = []

        # Primera simulación automática
        self.ejecutar_recalculo()

    def ejecutar_recalculo(self):
        try:
            ev0 = float(self.input_v0.text())
            ev1 = float(self.input_v1.text())
        except ValueError:
            QMessageBox.critical(self, "Error de formato", "Introduce números decimales válidos.")
            return

        x0  = -52409.924647197156 * 1000
        y0  = -48671.635720228245 * 1000
        vx0 = -1.22735334971968  * 1000 + ev0
        vy0 = -2.34643943970753  * 1000 + ev1
        estado_inicial = np.array([x0, y0, vx0, vy0])

        self.fuentes = resolver_fuentes(
            self.args.metodos, self.args.dt, self.args.duracion,
            estado_inicial, exportar_csv=self.args.exportar_csv
        )

        if not self.fuentes:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Sin datos válidos.", ha='center', va='center')
            self.canvas.draw()
            self.slider_tiempo.setEnabled(False)
            return

        max_pasos = max(len(df_orion) for (_, df_orion, _) in self.fuentes) - 1
        self.slider_tiempo.setMaximum(max_pasos)
        self.slider_tiempo.setValue(0)
        self.slider_tiempo.setEnabled(True)

        self.dibujar_grafico_base()

    def dibujar_grafico_base(self):
        self.ax.clear()
        self.marcadores_moviles.clear()

        # Dibujar la Tierra
        self.ax.add_patch(Circle((0, 0), EARTH_RADIUS_KM * 1e3, color='steelblue', zorder=5))
        self.ax.plot([], [], 's', color='steelblue', label='Tierra')

        # Dibujar las líneas completas de las órbitas
        for (etq, df_orion, df_luna), color in zip(self.fuentes, PALETA):
            self.ax.plot(df_orion['x'].values, df_orion['y'].values, lw=1.2, color=color, label=f'Orión ({etq})', zorder=4)

            if df_luna is not None:
                self.ax.plot(df_luna['x'].values, df_luna['y'].values, lw=1.0, color=color, ls='--', alpha=0.4, label=f'Luna ({etq})', zorder=3)

            # Inicializar los marcadores dinámicos que se moverán con el Slider
            marcador_p, = self.ax.plot([], [], 'o', color=color, ms=9, zorder=8)
            marcador_l = None
            if df_luna is not None:
                marcador_l, = self.ax.plot([], [], 'X', color=color, ms=8, zorder=8)
            
            self.marcadores_moviles.append((df_orion, df_luna, marcador_p, marcador_l))

        # Ajuste de límites por defecto de la ventana
        all_x = np.concatenate([df['x'].values for _, df, _ in self.fuentes])
        all_y = np.concatenate([df['y'].values for _, df, _ in self.fuentes])
        for _, _, df_luna in self.fuentes:
            if df_luna is not None:
                all_x = np.concatenate([all_x, df_luna['x'].values])
                all_y = np.concatenate([all_y, df_luna['y'].values])

        cx = (all_x.min() + all_x.max()) / 2
        cy = (all_y.min() + all_y.max()) / 2
        hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
        self.ax.set_xlim(cx - hs, cx + hs)
        self.ax.set_ylim(cy - hs, cy + hs)
        self.ax.set_aspect('equal', 'box')

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.legend(fontsize=8, loc='upper right')
        
        # NUEVO: Le dice a la barra de navegación que registre estos límites como la posición de inicio "Home"
        self.toolbar.update()

        self.texto_info = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, 
                                       fontsize=9, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        self.actualizar_marcador_tiempo(0)

    def actualizar_marcador_tiempo(self, paso):
        import datetime  # Importación local para el manejo del tiempo
        
        self.label_tiempo.setText(f"Paso: {paso} / {self.slider_tiempo.maximum()}")

        # Definimos una fecha base simulada para el inicio (puedes cambiarla por la real de la misión)
        fecha_inicio = datetime.datetime(2026, 1, 1, 0, 0, 0) 

        texto_lineas = []

        for df_orion, df_luna, marcador_p, marcador_l in self.marcadores_moviles:
            idx_p = min(paso, len(df_orion) - 1)
            
            # Extraer posición actual
            x_act = df_orion['x'].iloc[idx_p]
            y_act = df_orion['y'].iloc[idx_p]
            marcador_p.set_data([x_act], [y_act])

            if df_luna is not None and marcador_l is not None:
                idx_l = min(paso, len(df_luna) - 1)
                marcador_l.set_data([df_luna['x'].iloc[idx_l]], [df_luna['y'].iloc[idx_l]])

            # === NUEVO: Cálculos del Vector Velocidad y Tiempo ===
            vx = df_orion['vx'].iloc[idx_p]
            vy = df_orion['vy'].iloc[idx_p]
            t_segundos = df_orion['time_s'].iloc[idx_p]

            # 1. Magnitud de la velocidad: sqrt(vx^2 + vy^2)
            magnitud_v = np.hypot(vx, vy) # Equivale a np.sqrt(vx**2 + vy**2)

            # 2. Ángulo en grados respecto al eje X positivo (-180° a 180°)
            angulo_rad = np.arctan2(vy, vx)
            angulo_deg = np.degrees(angulo_rad)

            # 3. Calcular Fecha y Hora sumando los segundos transcurridos
            fecha_actual = fecha_inicio + datetime.timedelta(seconds=float(t_segundos))
            fecha_str = fecha_actual.strftime("%Y-%m-%d %H:%M:%S")

            # Formatear los datos para esta trayectoria
            # Nota: Si tienes múltiples integradores corriendo a la vez, esto listará los datos de cada uno
            texto_lineas.append(
                f"--- Orión ---\n"
                f"Fecha/Hora: {fecha_str}\n"
                f"Mag. Vel: {magnitud_v:.2f} m/s\n"
                f"Ángulo Vel: {angulo_deg:.2f}°"
            )

        # Actualizar el cuadro de texto en la esquina superior izquierda de la gráfica
        if self.texto_info:
            self.texto_info.set_text("\n".join(texto_lineas))

        # Usamos draw_idle() para que respete el zoom actual de la pantalla mientras mueves el slider
        self.canvas.draw_idle()


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Simulador interactivo con Zoom')
    p.add_argument('--metodos', nargs='+', required=True, help='Ej: rk4 euler')
    p.add_argument('--dt',       type=float, default=60.0)
    p.add_argument('--duracion', type=float, default=27.3)
    p.add_argument('--no-exportar-csv', dest='exportar_csv', action='store_false')
    return p.parse_args()

def main():
    args = parse_args()
    app = QApplication(sys.argv)
    ventana = VentanaOrbital(args)
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()