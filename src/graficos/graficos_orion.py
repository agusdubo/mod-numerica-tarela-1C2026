# Código hecho por: Agustín Dubovitsky Otero (padrón: 111954) y Tomás Bautista Conti (Padrón: 111760)
import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils.fisica import RADIO_TIERRA_KM

PALETA = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

FECHA_INICIO = datetime.datetime(2026, 4, 3,hour=5,minute=3)


def graficar_orbital(calcular_fuentes, ev0: float = -633, ev1: float = -36,
                     paso: int = 0, salida: str = "simulacion_orbital.png"):
    """Genera y guarda una imagen del simulador orbital.

    Parameters
    ----------
    calcular_fuentes : callable(ev0, ev1) -> list[(etq, df_orion, df_luna)]
        Función que recibe los deltas de velocidad inicial y devuelve las fuentes simuladas.
    ev0 : float
        Componente x de la velocidad extra inicial (m/s).
    ev1 : float
        Componente y de la velocidad extra inicial (m/s).
    paso : int
        Paso de tiempo a marcar con el indicador de posición (0 = inicio).
    salida : str
        Nombre del archivo de imagen a guardar (.png, .pdf, etc.).
    """
    fuentes = calcular_fuentes(ev0, ev1)

    fig, ax = plt.subplots(figsize=(7, 7))

    if not fuentes:
        ax.text(0.5, 0.5, "Sin datos válidos.", ha="center", va="center")
        fig.savefig(salida, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Imagen guardada en: {salida}")
        return

    marcadores = []

    # Tierra
    ax.add_patch(Circle((0, 0), RADIO_TIERRA_KM * 1_000, color="steelblue", zorder=5))
    ax.plot([], [], "s", color="steelblue", label="Tierra")

    for (etq, df_orion, df_luna), color in zip(fuentes, PALETA):
        # Trayectoria de Orión
        ax.plot(df_orion["x"].values, df_orion["y"].values,
                lw=1.2, color=color, label=f"Orión ({etq})", zorder=4)

        # Trayectoria de la Luna (si está disponible)
        if df_luna is not None:
            ax.plot(df_luna["x"].values, df_luna["y"].values,
                    lw=1.0, color=color, ls="--", alpha=0.4,
                    label=f"Luna ({etq})", zorder=3)

        marcador_orion, = ax.plot([], [], "o", color=color, ms=9, zorder=8)
        marcador_luna = None
        if df_luna is not None:
            marcador_luna, = ax.plot([], [], "X", color=color, ms=8, zorder=8)

        marcadores.append((df_orion, df_luna, marcador_orion, marcador_luna))

    # Ajuste de ejes
    all_x = np.concatenate([df["x"].values for _, df, _ in fuentes])
    all_y = np.concatenate([df["y"].values for _, df, _ in fuentes])
    for _, _, df_luna in fuentes:
        if df_luna is not None:
            all_x = np.concatenate([all_x, df_luna["x"].values])
            all_y = np.concatenate([all_y, df_luna["y"].values])

    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
    ax.set_xlim(cx - hs, cx + hs)
    ax.set_ylim(cy - hs, cy + hs)
    ax.set_aspect("equal", "box")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="upper right")

    
    lineas = []
    for df_orion, df_luna, marc_orion, marc_luna in marcadores:
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

    ax.text(
        0.02, 0.95, "\n".join(lineas),
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagen guardada en: {salida}")