#!/usr/bin/env python3
"""
Con este script se puede elegir un método con su paso y con su duración, para graficar la trayectoria de orión utilizando ese método.
Se grafica a partir de los resultados obtenidos con la simulación de la órbita lunar, es decir, si ejecuto este script con rk2, va a utilizar los resultados de la órbita lunar obtnidos con rk2.
Si no hay un csv en la carpeta resultados con el nombre del método, hay que correr primero el script para calcular la órbita lunar con el método para que se genere y ahí correr este.  

Uso:
    python trayectoria_orion.py --metodo rk4 
    python trayectoria_orion.py --metodo euler --dt 60 --duracion 27.3

"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSlider, QMessageBox)
from PyQt5.QtCore import Qt

from discretizaciones import INTEGRADORES

# ── Constantes físicas ───────────────────────────────────────────────────────
CONST_G = 6.674e-11
M_T     = 5.972e24
M_L     = 7.342e22
EARTH_RADIUS_KM = 6371
MOON_RADIUS_KM = 1737

PALETA = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']


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
        f = lambda t, s: derivadas(s, pos_luna)
        estado = integrador(f, t, estado, dt)
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

def resolver_fuentes(metodo, dt, duracion, estado_inicial):
    resultados = []
    if metodo.startswith('csv:'):
        ruta = metodo[4:]
        if os.path.exists(ruta):
            df = cargar_csv_trayectoria(ruta)
            resultados.append((os.path.splitext(os.path.basename(ruta))[0], df, None))
    elif metodo in INTEGRADORES:
        ruta_luna = f"resultados/resultado_{metodo.lower()}.csv"
        if os.path.exists(ruta_luna):
            datos_luna_arr, df_luna = cargar_datos_luna(ruta_luna)
            df = simular(metodo, dt, duracion, estado_inicial, datos_luna_arr)
            resultados.append((metodo.upper(), df, df_luna))
    return resultados

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

        layout_controles = QHBoxLayout()
        layout_controles.addWidget(QLabel("extra_v[0] (vx, m/s):"))
        self.input_v0 = QLineEdit("-633")
        self.input_v0.setFixedWidth(80)
        layout_controles.addWidget(self.input_v0)

        layout_controles.addWidget(QLabel("extra_v[1] (vy, m/s):"))
        self.input_v1 = QLineEdit("-36")
        self.input_v1.setFixedWidth(80)
        layout_controles.addWidget(self.input_v1)

        self.btn_recalcular = QPushButton("Recalcular")
        self.btn_recalcular.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")
        self.btn_recalcular.clicked.connect(self.ejecutar_recalculo)
        layout_controles.addWidget(self.btn_recalcular)
        layout_controles.addStretch()
        layout_principal.addLayout(layout_controles)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_principal.addWidget(self.toolbar)
        layout_principal.addWidget(self.canvas)

        
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

        self.ejecutar_recalculo()

    def ejecutar_recalculo(self):
        try:
            ev0 = float(self.input_v0.text())
            ev1 = float(self.input_v1.text())
        except ValueError:
            QMessageBox.critical(self, "Error de formato", "Introduce números decimales válidos.")
            return

        # Condiciones iniciales
        x0  = -52409.924647197156 * 1000
        y0  = -48671.635720228245 * 1000
        vx0 = -1.22735334971968  * 1000 + ev0
        vy0 = -2.34643943970753  * 1000 + ev1
        estado_inicial = np.array([x0, y0, vx0, vy0])

        self.fuentes = resolver_fuentes(
        self.args.metodo, self.args.dt, self.args.duracion, estado_inicial
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

        self.ax.add_patch(Circle((0, 0), EARTH_RADIUS_KM * 1e3, color='steelblue', zorder=5))
        self.ax.plot([], [], 's', color='steelblue', label='Tierra')

        for (etq, df_orion, df_luna), color in zip(self.fuentes, PALETA):
            self.ax.plot(df_orion['x'].values, df_orion['y'].values, lw=1.2, color=color, label=f'Orión ({etq})', zorder=4)

            if df_luna is not None:
                self.ax.plot(df_luna['x'].values, df_luna['y'].values, lw=1.0, color=color, ls='--', alpha=0.4, label=f'Luna ({etq})', zorder=3)

            marcador_p, = self.ax.plot([], [], 'o', color=color, ms=9, zorder=8)
            marcador_l = None
            if df_luna is not None:
                marcador_l, = self.ax.plot([], [], 'X', color=color, ms=8, zorder=8)
            
            self.marcadores_moviles.append((df_orion, df_luna, marcador_p, marcador_l))

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
        
        self.toolbar.update()

        self.texto_info = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, 
                                       fontsize=9, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        self.actualizar_marcador_tiempo(0)

    def actualizar_marcador_tiempo(self, paso):
        import datetime
        
        self.label_tiempo.setText(f"Paso: {paso} / {self.slider_tiempo.maximum()}")

        fecha_inicio = datetime.datetime(2026, 1, 1, 0, 0, 0) 

        texto_lineas = []

        for df_orion, df_luna, marcador_p, marcador_l in self.marcadores_moviles:
            idx_p = min(paso, len(df_orion) - 1)
            
            x_act = df_orion['x'].iloc[idx_p]
            y_act = df_orion['y'].iloc[idx_p]
            marcador_p.set_data([x_act], [y_act])

            if df_luna is not None and marcador_l is not None:
                idx_l = min(paso, len(df_luna) - 1)
                marcador_l.set_data([df_luna['x'].iloc[idx_l]], [df_luna['y'].iloc[idx_l]])

            vx = df_orion['vx'].iloc[idx_p]
            vy = df_orion['vy'].iloc[idx_p]
            t_segundos = df_orion['time_s'].iloc[idx_p]

            # Magnitud de la velocidad
            magnitud_v = np.hypot(vx, vy)

            # Ángulo en grados
            angulo_rad = np.arctan2(vy, vx)
            angulo_deg = np.degrees(angulo_rad)

            # Fecha y Hora
            fecha_actual = fecha_inicio + datetime.timedelta(seconds=float(t_segundos))
            fecha_str = fecha_actual.strftime("%Y-%m-%d %H:%M:%S")

            texto_lineas.append(
                f"--- Orión ---\n"
                f"Fecha/Hora: {fecha_str}\n"
                f"Mag. Vel: {magnitud_v:.2f} m/s\n"
                f"Ángulo Vel: {angulo_deg:.2f}°"
            )

        if self.texto_info:
            self.texto_info.set_text("\n".join(texto_lineas))

        self.canvas.draw_idle()


def parse_args():
    p = argparse.ArgumentParser(description='Simulador interactivo con Zoom')
    p.add_argument('--metodo', required=True, help='Ej: rk4')
    p.add_argument('--dt',       type=float, default=60.0)
    p.add_argument('--duracion', type=float, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    app = QApplication(sys.argv)
    ventana = VentanaOrbital(args)
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()