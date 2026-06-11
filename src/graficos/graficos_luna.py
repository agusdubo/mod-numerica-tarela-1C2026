import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils.fisica import RADIO_TIERRA_KM

PALETA = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def graficar_orbitas(fuentes: list[tuple[str, pd.DataFrame]],
                     salida: str,
                     dpi: int = 150) -> None:
    """Genera y guarda una imagen comparando las trayectorias recibidas

    fuentes : lista de (etiqueta, DataFrame con columnas x, y)
    salida  : ruta del archivo de imagen resultante
    dpi     : resolución de la imagen
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Tierra
    ax.add_patch(Circle((0, 0), RADIO_TIERRA_KM * 1_000,
                         color="steelblue", zorder=5))

    # Trayectorias
    for (etq, df), color in zip(fuentes, PALETA):
        ax.plot(df["x"].values, df["y"].values,
                lw=1.2, color=color, label=etq)

    all_x = np.concatenate([df["x"].values for _, df in fuentes])
    all_y = np.concatenate([df["y"].values for _, df in fuentes])
    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
    ax.set_xlim(cx - hs, cx + hs)
    ax.set_ylim(cy - hs, cy + hs)
    ax.set_aspect("equal", "box")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Comparación de trayectorias orbitales")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(salida, dpi=dpi, bbox_inches="tight")
    print(f"\n[graficos_luna] Imagen guardada: {salida}")
    plt.close(fig)