"""
Lectura y escritura de CSVs
"""

import os
import sys

import numpy as np
import pandas as pd


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a nombres canónicos (time_s, x, y, vx, vy)."""
    col_map = {}
    for col in df.columns:
        cu = col.strip().upper()
        if cu in ("JDTDB", "TIME_S", "TIME"):  col_map[col] = "time_s"
        elif cu in ("X", "X_LUNA"):             col_map[col] = "x"
        elif cu in ("Y", "Y_LUNA"):             col_map[col] = "y"
        elif cu == "VX":                        col_map[col] = "vx"
        elif cu == "VY":                        col_map[col] = "vy"
    return df.rename(columns=col_map)


def _convertir_a_metros(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte km a metros."""
    if df["x"].abs().max() < 1e7:
        df = df.copy()
        df["x"] *= 1_000
        df["y"] *= 1_000
    return df


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("x", "y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y"])


# ── Lectura ───────────────────────────────────────────────────────────────────
def cargar_csv_trayectoria(ruta: str) -> pd.DataFrame:
    """Lee un CSV de (propio o Horizons).
    Devuelve DataFrame con columnas [time_s, x, y, ...] en metros."""
    with open(ruta) as f:
        lineas = f.readlines()

    if any(";" in l for l in lineas[:5]):
        df = pd.read_csv(ruta, sep=";", decimal=",", header=None,
                         names=["time_s", "x", "y", "z", "vx", "vy", "vz"])
    else:
        # Buscar la fila de encabezado con columnas X e Y
        header_idx = next(
            (i for i, l in enumerate(lineas)
             if {"X", "Y"} <= {p.strip().upper() for p in l.split(",")}),
            None,
        )
        if header_idx is None:
            print(f"Error: no se encontró encabezado con X e Y en '{ruta}'",
                  file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(ruta, sep=",", header=header_idx, low_memory=False)

    df = _normalizar_columnas(df)
    df = _limpiar(df)
    df = _convertir_a_metros(df)

    print(f"  [CSV] {os.path.basename(ruta)} cargado ({len(df):,} filas)")
    return df


def cargar_posiciones_luna(ruta: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Carga un CSV de órbita lunar generado por trayectoria_luna.py.
    Devuelve (array de posiciones, DataFrame completo)."""
    df = cargar_csv_trayectoria(ruta)
    return df[["x", "y"]].to_numpy(), df[["x", "y"]].reset_index(drop=True)


# ── Escritura ─────────────────────────────────────────────────────────────────
def guardar_csv(df: pd.DataFrame, ruta: str) -> None:
    """Guarda un DataFrame en CSV, creando la carpeta si no existe."""
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"  Resultados guardados en: {ruta}")