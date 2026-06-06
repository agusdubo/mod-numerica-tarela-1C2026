#!/usr/bin/env python3
"""
Graficador interactivo de trayectorias espaciales cargadas desde archivos CSV en PyQt5.
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
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Importar componentes de PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QMessageBox)
from PyQt5.QtCore import Qt

# ── Constantes físicas ───────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371
PALETA = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

# ────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS DESDE CSV
# ────────────────────────────────────────────────────────────────────────────
def cargar_csv_trayectoria(ruta: str) -> pd.DataFrame:
    """Carga un archivo CSV buscando de manera flexible las columnas X e Y."""
    if not os.path.exists(ruta):
        print(f"Error: No se encontró el archivo {ruta}")
        return None

    with open(ruta) as f: 
        lineas = f.readlines()
    
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
        if header_idx is None:
            # Si no encuentra cabecera explícita, intenta leer asumiendo que la primera fila es válida
            header_idx = 0
            
        df = pd.read_csv(ruta, sep=',', header=header_idx, low_memory=False)
        
        col_map = {}
        for col in df.columns:
            cu = str(col).strip().upper()
            if cu in ('JDTDB', 'TIME_S', 'TIME', 'TIEMPO'): col_map[col] = 'time_s'
            elif cu in ('X', 'X_LUNA', 'POS_X'):  col_map[col] = 'x'
            elif cu in ('Y', 'Y_LUNA', 'POS_Y'):  col_map[col] = 'y'
        df = df.rename(columns=col_map)
        
    for c in ('x', 'y'): 
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=['x', 'y'])
    
    # Conversión automática a metros si los datos vienen en kilómetros
    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3
    return df.reset_index(drop=True)


def procesar_fuentes_csv(archivos_capsula):
    """
    Busca los archivos de la cápsula y sus equivalentes de la luna.
    Ejemplo: si pasas 'orion_mision1.csv', buscará 'orion_mision1_luna.csv' o 'resultado_luna.csv'
    """
    resultados = []
    for ruta in archivos_capsula:
        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            continue
            
        df_capsula = cargar_csv_trayectoria(ruta)
        if df_capsula is None or df_capsula.empty:
            continue
            
        nombre_base = os.path.splitext(os.path.basename(ruta))[0]
        
        # Intentar buscar un archivo de luna asociado automáticamente
        # Opción A: nombre_base + _luna.csv (ej: orion1_luna.csv)
        # Opción B: resultado_luna.csv (por compatibilidad con tu esquema anterior)
        ruta_luna_a = ruta.replace(".csv", "_luna.csv")
        ruta_luna_b = os.path.join(os.path.dirname(ruta), "resultado_luna.csv")
        
        df_luna = None
        if os.path.exists(ruta_luna_a):
            df_luna = cargar_csv_trayectoria(ruta_luna_a)
        elif os.path.exists(ruta_luna_b):
            df_luna = cargar_csv_trayectoria(ruta_luna_b)
            
        resultados.append((nombre_base, df_capsula, df_luna))
        
    return resultados


# ────────────────────────────────────────────────────────────────────────────
# INTERFAZ GRÁFICA EN PYQT5
# ────────────────────────────────────────────────────────────────────────────
class VentanaOrbital(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fuentes = []
        
        self.setWindowTitle("Visualizador de Trayectorias desde CSV")
        self.resize(950, 850)

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # --- Lienzo del Gráfico ---
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        
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

        # Cargar y graficar datos
        self.cargar_y_dibujar()

    def cargar_y_dibujar(self):
        self.fuentes = procesar_fuentes_csv(self.args.archivos)

        if not self.fuentes:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No se pudieron cargar archivos válidos.", ha='center', va='center')
            self.canvas.draw()
            self.slider_tiempo.setEnabled(False)
            return

        # Encontrar el archivo con más filas para limitar el slider general
        max_pasos = max(len(df_capsula) for (_, df_capsula, _) in self.fuentes) - 1
        self.slider_tiempo.setMaximum(max_pasos)
        self.slider_tiempo.setValue(0)
        self.slider_tiempo.setEnabled(True)

        self.dibujar_grafico_base()

    def dibujar_grafico_base(self):
        self.ax.clear()
        self.marcadores_moviles.clear()

        # Dibujar la Tierra en el origen (0,0)
        self.ax.add_patch(Circle((0, 0), EARTH_RADIUS_KM * 1e3, color='steelblue', zorder=5))
        self.ax.plot([], [], 's', color='steelblue', label='Tierra')

        # Dibujar las líneas completas de las órbitas cargadas
        for (etq, df_capsula, df_luna), color in zip(self.fuentes, PALETA):
            # Dibujar Cápsula
            self.ax.plot(df_capsula['x'].values, df_capsula['y'].values, lw=1.2, color=color, label=f'Cápsula ({etq})', zorder=4)

            # Dibujar Luna si existe su archivo
            if df_luna is not None:
                self.ax.plot(df_luna['x'].values, df_luna['y'].values, lw=1.0, color=color, ls='--', alpha=0.4, label=f'Luna ({etq})', zorder=3)

            # Inicializar los marcadores dinámicos para el Slider
            marcador_p, = self.ax.plot([], [], 'o', color=color, ms=9, zorder=8)
            marcador_l = None
            if df_luna is not None:
                marcador_l, = self.ax.plot([], [], 'X', color=color, ms=8, zorder=8)
            
            self.marcadores_moviles.append((df_capsula, df_luna, marcador_p, marcador_l))

        # Ajuste dinámico de los límites de los ejes (Bounding Box)
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
        self.actualizar_marcador_tiempo(0)

    def actualizar_marcador_tiempo(self, paso):
        self.label_tiempo.setText(f"Paso: {paso} / {self.slider_tiempo.maximum()}")

        for df_capsula, df_luna, marcador_p, marcador_l in self.marcadores_moviles:
            # Controlar que el índice no supere el tamaño del DataFrame específico
            idx_p = min(paso, len(df_capsula) - 1)
            marcador_p.set_data([df_capsula['x'].iloc[idx_p]], [df_capsula['y'].iloc[idx_p]])

            if df_luna is not None and marcador_l is not None:
                idx_l = min(paso, len(df_luna) - 1)
                marcador_l.set_data([df_luna['x'].iloc[idx_l]], [df_luna['y'].iloc[idx_l]])

        self.canvas.draw_idle()


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Visualizador interactivo de trayectorias espaciales desde archivos CSV')
    p.add_argument('--archivos', nargs='+', required=True, help='Lista de archivos CSV de las cápsulas (ej: datos_orion.csv)')
    return p.parse_args()

def main():
    args = parse_args()
    app = QApplication(sys.argv)
    ventana = VentanaOrbital(args)
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()