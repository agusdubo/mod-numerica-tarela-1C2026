#!/usr/bin/env python3
"""
Comparador de trayectorias orbitales lunares
Métodos disponibles: rk2 | rk4 | euler | csv:lunar_orbit.csv

Uso:
    python comparar.py --metodos rk4 
    python comparar.py --metodos rk4 euler
    python comparar.py --metodos rk4 euler rk2
    python comparar.py --metodos rk4 euler rk2 csv:lunar_orbit.csv
    python comparar.py --metodos rk4 euler --dt 60 --duracion 27.3
    python comparar.py --metodos rk4 euler --salida comparacion.png

    SIENDO dt paso del tiempo y duracion los dias
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from integradores import INTEGRADORES


# ── Constantes físicas ───────────────────────────────────────────────────────
G   = 6.674e-11
M_T = 5.972e24

R_PERIGEO = 3.565e8
R_APOGEO  = 4.067e8
A_SEMIEJE = (R_PERIGEO + R_APOGEO) / 2
V_PERIGEO = np.sqrt(G * M_T * (2 / R_PERIGEO - 1 / A_SEMIEJE))

EARTH_RADIUS_KM = 6371

PALETA = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#9467bd',
    '#8c564b',
    '#e377c2',
]


# ────────────────────────────────────────────────────────────────────────────
# FÍSICA
# ────────────────────────────────────────────────────────────────────────────

def derivadas(t, estado):
    x, y, vx, vy = estado
    d2 = x**2 + y**2
    d  = np.sqrt(d2)
    ax = -G * M_T / d2 * (x / d)
    ay = -G * M_T / d2 * (y / d)
    return np.array([vx, vy, ax, ay])


# ────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN
# ────────────────────────────────────────────────────────────────────────────

def simular(metodo: str, dt: float, duracion_dias: float) -> pd.DataFrame:
    integrador = INTEGRADORES[metodo]
    t_total = duracion_dias * 86400.0
    pasos   = int(t_total / dt)

    estado = np.array([R_PERIGEO, 0.0, 0.0, V_PERIGEO])
    t      = 0.0

    registros = np.empty((pasos + 1, 5))
    registros[0] = [t, estado[0], estado[1], estado[2], estado[3]]

    print(f"  [{metodo.upper()}] dt={dt}s | {duracion_dias} días | {pasos:,} pasos...", end=' ')

    for i in range(pasos):
        estado = integrador(derivadas, t, estado, dt)
        t     += dt
        registros[i + 1] = [t, estado[0], estado[1], estado[2], estado[3]]

    df = pd.DataFrame(registros, columns=['time_s', 'x', 'y', 'vx', 'vy'])
    print(f"listo ({len(df):,} filas)")
    return df


# ────────────────────────────────────────────────────────────────────────────
# CARGA DE CSV
# ────────────────────────────────────────────────────────────────────────────

def cargar_csv(ruta: str) -> pd.DataFrame:
    with open(ruta, 'r') as f:
        lineas = f.readlines()

    sep = ';' if any(';' in l for l in lineas[:5]) else ','

    if sep == ';':
        df = pd.read_csv(ruta, sep=';', decimal=',', header=None,
                         names=['time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    else:
        # Buscar la fila del encabezado: la primera que contenga 'X' e 'Y'
        header_idx = None
        for i, linea in enumerate(lineas):
            partes = [p.strip().upper() for p in linea.split(',')]
            if 'X' in partes and 'Y' in partes:
                header_idx = i
                break

        if header_idx is None:
            print(f"Error: no se encontró encabezado con columnas X e Y en '{ruta}'",
                  file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(ruta, sep=',', header=header_idx, low_memory=False)

        # Descartar filas que no sean datos numéricos (pie de página de Horizons, etc.)
        col_map = {}
        for col in df.columns:
            cu = col.strip().upper()
            if cu in ('JDTDB', 'TIME_S', 'TIME'): col_map[col] = 'time_s'
            elif cu == 'X':                        col_map[col] = 'x'
            elif cu == 'Y':                        col_map[col] = 'y'
            elif cu == 'VX':                       col_map[col] = 'vx'
            elif cu == 'VY':                       col_map[col] = 'vy'
        df = df.rename(columns=col_map)

    for c in ['x', 'y']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['x', 'y'])

    # Las coordenadas de Horizons vienen en km → convertir a metros
    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3

    print(f"  [CSV] {os.path.basename(ruta)} cargado ({len(df):,} filas)")
    return df


# ────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN DE FUENTES
# ────────────────────────────────────────────────────────────────────────────

def resolver_fuentes(metodos, dt, duracion, exportar_csv=True):
    resultados = []
    for m in metodos:
        if m.startswith('csv:'):
            ruta = m[4:]
            if not os.path.exists(ruta):
                print(f"Error: no se encontró '{ruta}'", file=sys.stderr)
                sys.exit(1)
            df  = cargar_csv(ruta)
            etq = os.path.splitext(os.path.basename(ruta))[0]
            # Los CSV de entrada no se re-exportan (ya son archivos externos)
        elif m in INTEGRADORES:
            df  = simular(m, dt, duracion)
            etq = m.upper()
            if exportar_csv:
                nombre_csv = f"resultado_{etq.lower()}.csv"
                df.to_csv(nombre_csv, index=False)
                print(f"  [{etq}] Resultados guardados en: {nombre_csv}")
        else:
            print(f"Error: método desconocido '{m}'. Opciones: rk4, euler, csv:<ruta>",
                  file=sys.stderr)
            sys.exit(1)
        resultados.append((etq, df))
    return resultados


# ────────────────────────────────────────────────────────────────────────────
# GRAFICACIÓN
# ────────────────────────────────────────────────────────────────────────────

def graficar(fuentes, salida, dpi):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Tierra
    ax.add_patch(Circle((0, 0), EARTH_RADIUS_KM * 1e3,
                         color='steelblue', zorder=5))

    # Trayectorias
    for (etq, df), color in zip(fuentes, PALETA):
        ax.plot(df['x'].values, df['y'].values,
                lw=1.2, color=color, label=etq)

    # Límites cuadrados
    all_x = np.concatenate([df['x'].values for _, df in fuentes])
    all_y = np.concatenate([df['y'].values for _, df in fuentes])
    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
    ax.set_xlim(cx - hs, cx + hs)
    ax.set_ylim(cy - hs, cy + hs)
    ax.set_aspect('equal', 'box')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Comparación de trayectorias orbitales')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(salida, dpi=dpi, bbox_inches='tight')
    print(f"\n[comparar_orbitas] Imagen guardada: {salida}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compara trayectorias orbitales lunares.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument(
        '--metodos', nargs='+', required=True,
        metavar='MÉTODO',
        help='Uno o más métodos: rk4, euler, csv:<ruta>. Ej: --metodos rk4 euler csv:luna.csv'
    )
    p.add_argument('--dt',       type=float, default=60.0,
                   help='Paso de tiempo en segundos (default: 60)')
    p.add_argument('--duracion', type=float, default=27.3,
                   help='Duración en días (default: 27.3)')
    p.add_argument('--salida',   type=str,   default='comparacion_orbital.png',
                   help='Ruta de la imagen de salida (default: comparacion_orbital.png)')
    p.add_argument('--dpi',      type=int,   default=150,
                   help='Resolución de la imagen (default: 150)')
    p.add_argument('--no-exportar-csv', dest='exportar_csv', action='store_false',
                   help='Desactiva la exportación de CSV por método (activada por defecto)')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[comparar_orbitas] Fuentes: {args.metodos}")
    fuentes = resolver_fuentes(args.metodos, args.dt, args.duracion,
                               exportar_csv=args.exportar_csv)
    graficar(fuentes, salida=args.salida, dpi=args.dpi)


if __name__ == '__main__':
    main()