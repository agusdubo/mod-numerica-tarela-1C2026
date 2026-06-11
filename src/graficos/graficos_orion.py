import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QSlider, QMessageBox,
)
from PyQt5.QtCore import Qt

from utils.fisica import RADIO_TIERRA_KM

PALETA = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

FECHA_INICIO = datetime.datetime(2026, 1, 1)


class VentanaOrbital(QMainWindow):
    """Ventana principal del simulador de Orión.

    calcular_fuentes : callable(ev0, ev1) -> list[(etq, df_orion, df_luna)]
        Función que recibe los deltas de velocidad inicial y devuelve las fuentes ya simuladas
    """

    def __init__(self, calcular_fuentes):
        super().__init__()
        self._calcular_fuentes = calcular_fuentes
        self._fuentes          = []
        self._marcadores       = []

        self.setWindowTitle("Simulador Orbital Interactivo (Zoom Habilitado)")
        self.resize(950, 850)

        # ── Layout principal ──────────────────────────────────────────────────
        widget_central  = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        self.setCentralWidget(widget_central)

        # Controles de velocidad inicial
        layout_controles = QHBoxLayout()
        layout_controles.addWidget(QLabel("extra_v[0] (vx, m/s):"))
        self._input_v0 = QLineEdit("-633")
        self._input_v0.setFixedWidth(80)
        layout_controles.addWidget(self._input_v0)

        layout_controles.addWidget(QLabel("extra_v[1] (vy, m/s):"))
        self._input_v1 = QLineEdit("-36")
        self._input_v1.setFixedWidth(80)
        layout_controles.addWidget(self._input_v1)

        btn = QPushButton("Recalcular")
        btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")
        btn.clicked.connect(self._recalcular)
        layout_controles.addWidget(btn)
        layout_controles.addStretch()
        layout_principal.addLayout(layout_controles)

        # Canvas de matplotlib
        self._fig, self._ax = plt.subplots(figsize=(7, 7))
        self._canvas  = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout_principal.addWidget(self._toolbar)
        layout_principal.addWidget(self._canvas)

        # Slider de tiempo
        layout_slider = QHBoxLayout()
        self._label_paso = QLabel("Paso: 0 / 0")
        self._label_paso.setFixedWidth(150)
        layout_slider.addWidget(self._label_paso)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._actualizar_marcador)
        layout_slider.addWidget(self._slider)
        layout_principal.addLayout(layout_slider)

        self._texto_info = None
        self._recalcular()

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _recalcular(self):
        try:
            ev0 = float(self._input_v0.text())
            ev1 = float(self._input_v1.text())
        except ValueError:
            QMessageBox.critical(self, "Error de formato", "Introducí números decimales válidos.")
            return

        self._fuentes = self._calcular_fuentes(ev0, ev1)

        if not self._fuentes:
            self._ax.clear()
            self._ax.text(0.5, 0.5, "Sin datos válidos.", ha="center", va="center")
            self._canvas.draw()
            self._slider.setEnabled(False)
            return

        max_pasos = max(len(df) for _, df, _ in self._fuentes) - 1
        self._slider.setMaximum(max_pasos)
        self._slider.setValue(0)
        self._slider.setEnabled(True)
        self._dibujar_base()

    def _dibujar_base(self):
        self._ax.clear()
        self._marcadores.clear()

        # Tierra
        self._ax.add_patch(Circle((0, 0), RADIO_TIERRA_KM * 1_000, color="steelblue", zorder=5))
        self._ax.plot([], [], "s", color="steelblue", label="Tierra")

        for (etq, df_orion, df_luna), color in zip(self._fuentes, PALETA):
            # Trayectoria de Orión
            self._ax.plot(df_orion["x"].values, df_orion["y"].values,
                          lw=1.2, color=color, label=f"Orión ({etq})", zorder=4)
            # Trayectoria de la Luna (si está disponible)
            if df_luna is not None:
                self._ax.plot(df_luna["x"].values, df_luna["y"].values,
                              lw=1.0, color=color, ls="--", alpha=0.4,
                              label=f"Luna ({etq})", zorder=3)

            marcador_orion, = self._ax.plot([], [], "o", color=color, ms=9, zorder=8)
            marcador_luna   = None
            if df_luna is not None:
                marcador_luna, = self._ax.plot([], [], "X", color=color, ms=8, zorder=8)

            self._marcadores.append((df_orion, df_luna, marcador_orion, marcador_luna))

        all_x = np.concatenate([df["x"].values for _, df, _ in self._fuentes])
        all_y = np.concatenate([df["y"].values for _, df, _ in self._fuentes])
        for _, _, df_luna in self._fuentes:
            if df_luna is not None:
                all_x = np.concatenate([all_x, df_luna["x"].values])
                all_y = np.concatenate([all_y, df_luna["y"].values])

        cx = (all_x.min() + all_x.max()) / 2
        cy = (all_y.min() + all_y.max()) / 2
        hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
        self._ax.set_xlim(cx - hs, cx + hs)
        self._ax.set_ylim(cy - hs, cy + hs)
        self._ax.set_aspect("equal", "box")

        self._ax.set_xlabel("x (m)")
        self._ax.set_ylabel("y (m)")
        self._ax.legend(fontsize=8, loc="upper right")
        self._toolbar.update()

        self._texto_info = self._ax.text(
            0.02, 0.95, "",
            transform=self._ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        self._actualizar_marcador(0)

    def _actualizar_marcador(self, paso: int):
        self._label_paso.setText(f"Paso: {paso} / {self._slider.maximum()}")
        lineas = []

        for df_orion, df_luna, marc_orion, marc_luna in self._marcadores:
            idx = min(paso, len(df_orion) - 1)

            marc_orion.set_data([df_orion["x"].iloc[idx]],
                                [df_orion["y"].iloc[idx]])
            if df_luna is not None and marc_luna is not None:
                idx_l = min(paso, len(df_luna) - 1)
                marc_luna.set_data([df_luna["x"].iloc[idx_l]],
                                   [df_luna["y"].iloc[idx_l]])

            vx  = df_orion["vx"].iloc[idx]
            vy  = df_orion["vy"].iloc[idx]
            t_s = df_orion["time_s"].iloc[idx]

            mag_v      = np.hypot(vx, vy)
            angulo_deg = np.degrees(np.arctan2(vy, vx))
            fecha_str  = (FECHA_INICIO + datetime.timedelta(seconds=float(t_s))
                          ).strftime("%Y-%m-%d %H:%M:%S")

            lineas.append(
                f"--- Orión ---\n"
                f"Fecha/Hora: {fecha_str}\n"
                f"Mag. Vel: {mag_v:.2f} m/s\n"
                f"Ángulo Vel: {angulo_deg:.2f}°"
            )

        if self._texto_info:
            self._texto_info.set_text("\n".join(lineas))

        self._canvas.draw_idle()