#!/usr/bin/env python3
"""
Comparador de trayectorias orbitales con gravedad Tierra + Luna variable.
La posición de la Luna se lee de un CSV externo (datos_luna.csv por defecto).

Métodos disponibles: euler | rk2 | rk4 | csv:<ruta>

Uso:
    python comparar_artemis.py --metodos rk4
    python comparar_artemis.py --metodos rk4 euler rk2
    python comparar_artemis.py --metodos rk4 euler csv:referencia.csv
    python comparar_artemis.py --metodos rk4 --dt 60 --duracion 27.3
    python comparar_artemis.py --metodos rk4 euler --salida comparacion.png
    python comparar_artemis.py --metodos rk4 --no-exportar-csv

    SIENDO dt paso del tiempo (segundos) y duracion en días.
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


# ── Constantes físicas ───────────────────────────────────────────────────────
CONST_G = 6.674e-11
M_T     = 5.972e24
M_L     = 7.342e22

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

def gravitational_acceleration(x, y, pos_luna):
    """
    Aceleración gravitacional total (Tierra + Luna) en el punto (x, y).

    Parámetros
    ----------
    x, y     : coordenadas del objeto a integrar (m).
    pos_luna : tupla/array (x_luna, y_luna) con la posición de la Luna (m).

    Retorna
    -------
    (ax, ay) : componentes de la aceleración total (m/s²).
    """
    # — Tierra —
    r2_t = x**2 + y**2
    r_t  = r2_t**0.5
    if r_t == 0:
        ax_t = ay_t = 0.0
    else:
        a_t  = CONST_G * M_T / r2_t
        ax_t = -a_t * (x / r_t)
        ay_t = -a_t * (y / r_t)

    # — Luna —
    dx   = x - pos_luna[0]
    dy   = y - pos_luna[1]
    r2_l = dx**2 + dy**2
    r_l  = r2_l**0.5
    if r_l == 0:
        ax_l = ay_l = 0.0
    else:
        a_l  = CONST_G * M_L / r2_l
        ax_l = -a_l * (dx / r_l)
        ay_l = -a_l * (dy / r_l)

    return (ax_t + ax_l, ay_t + ay_l)


def derivadas(estado, pos_luna):
    """
    Devuelve [vx, vy, ax, ay] dado el estado [x, y, vx, vy] y la posición lunar.
    """
    x, y, vx, vy = estado
    ax, ay = gravitational_acceleration(x, y, pos_luna)
    return np.array([vx, vy, ax, ay])


# ────────────────────────────────────────────────────────────────────────────
# INTEGRADORES  (todos reciben estado, dt, pos_luna)
# ────────────────────────────────────────────────────────────────────────────

def euler_step(estado, dt, pos_luna):
    return estado + dt * derivadas(estado, pos_luna)


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


INTEGRADORES = {
    'euler': euler_step,
    'rk2':   rk2_step,
    'rk4':   rk4_step,
}


# ────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS LUNARES
# ────────────────────────────────────────────────────────────────────────────

def cargar_datos_luna(ruta: str) -> np.ndarray:
    """
    Carga las posiciones de la Luna desde un CSV.
    Espera columnas x_luna, y_luna (o X, Y) en metros.
    Si los valores parecen estar en km los convierte a metros.

    Retorna array de forma (N, 2).
    """
    df = pd.read_csv(ruta, low_memory=False)

    # Normalizar nombres de columna
    col_map = {}
    for col in df.columns:
        cu = col.strip().upper()
        if cu in ('X_LUNA', 'X'):  col_map[col] = 'x'
        elif cu in ('Y_LUNA', 'Y'): col_map[col] = 'y'
    df = df.rename(columns=col_map)

    for c in ('x', 'y'):
        if c not in df.columns:
            print(f"Error: columna '{c}' no encontrada en '{ruta}'", file=sys.stderr)
            sys.exit(1)
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['x', 'y'])

    # Horizons entrega km → convertir si los valores son pequeños
    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3

    print(f"  [LUNA] {os.path.basename(ruta)} cargado ({len(df):,} filas)")
    return df[['x', 'y']].to_numpy(), df[['x', 'y']].reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN
# ────────────────────────────────────────────────────────────────────────────

def simular(nombre: str, dt: float, duracion_dias: float,
            estado_inicial: np.ndarray, datos_luna: np.ndarray) -> pd.DataFrame:
    """
    Integra la trayectoria usando el método indicado.

    La posición lunar en el paso i se obtiene de datos_luna[i].
    Si la simulación tiene más pasos que filas en datos_luna, el último
    valor conocido se repite (advertencia emitida una sola vez).
    """
    integrador = INTEGRADORES[nombre]
    pasos      = int(duracion_dias * 86400.0 / dt)
    n_luna     = len(datos_luna)

    estado = estado_inicial.copy()
    t      = 0.0

    registros = np.empty((pasos + 1, 5))
    registros[0] = [t, *estado]

    advertido = False
    print(f"  [{nombre.upper()}] dt={dt}s | {duracion_dias} días | {pasos:,} pasos...", end=' ')

    for i in range(pasos):
        if i >= n_luna:
            if not advertido:
                print(f"\n  Advertencia: datos_luna agotados en paso {i}, "
                      f"repitiendo último valor.", file=sys.stderr)
                advertido = True
            pos_luna = datos_luna[-1]
        else:
            pos_luna = datos_luna[i]

        estado = integrador(estado, dt, pos_luna)
        t     += dt
        registros[i + 1] = [t, *estado]

    df = pd.DataFrame(registros, columns=['time_s', 'x', 'y', 'vx', 'vy'])
    print(f"listo ({len(df):,} filas)")
    return df


# ────────────────────────────────────────────────────────────────────────────
# CARGA DE CSV DE TRAYECTORIA (fuente externa)
# ────────────────────────────────────────────────────────────────────────────

def cargar_csv_trayectoria(ruta: str) -> pd.DataFrame:
    with open(ruta) as f:
        lineas = f.readlines()

    sep = ';' if any(';' in l for l in lineas[:5]) else ','

    if sep == ';':
        df = pd.read_csv(ruta, sep=';', decimal=',', header=None,
                         names=['time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    else:
        header_idx = None
        for i, linea in enumerate(lineas):
            partes = [p.strip().upper() for p in linea.split(',')]
            if 'X' in partes and 'Y' in partes:
                header_idx = i
                break
        if header_idx is None:
            print(f"Error: sin encabezado X/Y en '{ruta}'", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(ruta, sep=',', header=header_idx, low_memory=False)

        col_map = {}
        for col in df.columns:
            cu = col.strip().upper()
            if cu in ('JDTDB', 'TIME_S', 'TIME'): col_map[col] = 'time_s'
            elif cu == 'X':  col_map[col] = 'x'
            elif cu == 'Y':  col_map[col] = 'y'
            elif cu == 'VX': col_map[col] = 'vx'
            elif cu == 'VY': col_map[col] = 'vy'
        df = df.rename(columns=col_map)

    for c in ('x', 'y'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['x', 'y'])

    if df['x'].abs().max() < 1e7:
        df['x'] *= 1e3
        df['y'] *= 1e3

    print(f"  [CSV] {os.path.basename(ruta)} cargado ({len(df):,} filas)")
    return df


# ────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN DE FUENTES
# ────────────────────────────────────────────────────────────────────────────

def resolver_fuentes(metodos, dt, duracion, estado_inicial, exportar_csv=True):
    # Retorna lista de (etiqueta, df_orion, df_luna_o_None)
    resultados = []
    for m in metodos:
        if m.startswith('csv:'):
            ruta = m[4:]
            if not os.path.exists(ruta):
                print(f"Error: no se encontró '{ruta}'", file=sys.stderr)
                sys.exit(1)
            df  = cargar_csv_trayectoria(ruta)
            etq = os.path.splitext(os.path.basename(ruta))[0]
            resultados.append((etq, df, None))

        elif m in INTEGRADORES:
            ruta_luna = f"resultado_{m.lower()}.csv"
            if not os.path.exists(ruta_luna):
                print(f"Error: no se encontró '{ruta_luna}' para el método '{m}'.\n"
                      f"  Generalo primero con comparar.py --metodos {m}",
                      file=sys.stderr)
                sys.exit(1)
            datos_luna_arr, df_luna = cargar_datos_luna(ruta_luna)

            df  = simular(m, dt, duracion, estado_inicial, datos_luna_arr)
            etq = m.upper()
            if exportar_csv:
                nombre_csv = f"trayectoria_{etq.lower()}.csv"
                df.to_csv(nombre_csv, index=False)
                print(f"  [{etq}] Trayectoria guardada en: {nombre_csv}")

            resultados.append((etq, df, df_luna))

        else:
            print(f"Error: método desconocido '{m}'. Opciones: euler, rk2, rk4, csv:<ruta>",
                  file=sys.stderr)
            sys.exit(1)

    return resultados


# ────────────────────────────────────────────────────────────────────────────
# GRAFICACIÓN
# ────────────────────────────────────────────────────────────────────────────

MOON_RADIUS_KM = 1737  # radio lunar en km

def graficar(fuentes, salida, dpi):
    fig, ax = plt.subplots(figsize=(9, 9))

    # — Tierra —
    ax.add_patch(Circle((0, 0), EARTH_RADIUS_KM * 1e3,
                         color='steelblue', zorder=5))
    ax.plot([], [], 's', color='steelblue', label='Tierra')

    # — Trayectorias de Orión y Luna —
    for (etq, df_orion, df_luna), color in zip(fuentes, PALETA):
        # Trayectoria de Orión
        ax.plot(df_orion['x'].values, df_orion['y'].values,
                lw=1.2, color=color, label=f'Orión ({etq})', zorder=4)

        # Punto de inicio de Orión
        ax.plot(df_orion['x'].iloc[0], df_orion['y'].iloc[0],
                'o', color=color, ms=6, zorder=7,
                label=f'Inicio Orión ({etq})')

        # Trayectoria y posición final de la Luna (si existe)
        if df_luna is not None:
            ax.plot(df_luna['x'].values, df_luna['y'].values,
                    lw=1.0, color=color, ls='--', alpha=0.5,
                    label=f'Luna ({etq})', zorder=3)

            # Disco lunar en la posición final
            x_luna_fin = df_luna['x'].iloc[-1]
            y_luna_fin = df_luna['y'].iloc[-1]
            ax.add_patch(Circle((x_luna_fin, y_luna_fin),
                                 MOON_RADIUS_KM * 1e3,
                                 color=color, alpha=0.6, zorder=6))

    # — Límites cuadrados que incluyen todo —
    all_x = np.concatenate([df['x'].values for _, df, _ in fuentes])
    all_y = np.concatenate([df['y'].values for _, df, _ in fuentes])
    for _, _, df_luna in fuentes:
        if df_luna is not None:
            all_x = np.concatenate([all_x, df_luna['x'].values])
            all_y = np.concatenate([all_y, df_luna['y'].values])

    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    hs = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
    ax.set_xlim(cx - hs, cx + hs)
    ax.set_ylim(cy - hs, cy + hs)
    ax.set_aspect('equal', 'box')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Trayectorias Orión + Luna (Tierra + Luna)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(salida, dpi=dpi, bbox_inches='tight')
    print(f"\n[comparar_artemis] Imagen guardada: {salida}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compara trayectorias orbitales con gravedad Tierra + Luna variable.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--metodos', nargs='+', required=True, metavar='MÉTODO',
                   help='Uno o más de: euler, rk2, rk4, csv:<ruta>')
    p.add_argument('--dt',       type=float, default=60.0,
                   help='Paso de tiempo en segundos (default: 60)')
    p.add_argument('--duracion', type=float, default=27.3,
                   help='Duración en días (default: 27.3)')
    p.add_argument('--x0',  type=float, default=None,
                   help='Posición inicial x (m). Por defecto: perigeo lunar aprox.')
    p.add_argument('--y0',  type=float, default=0.0,   help='Posición inicial y (m)')
    p.add_argument('--vx0', type=float, default=0.0,   help='Velocidad inicial vx (m/s)')
    p.add_argument('--vy0', type=float, default=None,
                   help='Velocidad inicial vy (m/s). Por defecto: velocidad circular')
    p.add_argument('--salida', type=str, default='comparacion_artemis.png',
                   help='Ruta de la imagen de salida (default: comparacion_artemis.png)')
    p.add_argument('--dpi',    type=int, default=150,
                   help='Resolución de la imagen (default: 150)')
    p.add_argument('--no-exportar-csv', dest='exportar_csv', action='store_false',
                   help='Desactiva la exportación de CSV por método')
    return p.parse_args()


def main():
    args = parse_args()

    # — Estado inicial —
    # Valores por defecto: perigeo de órbita lunar media
    R_PERIGEO = 3.565e8
    V_PERIGEO = np.sqrt(CONST_G * M_T * (2 / R_PERIGEO - 1 / ((R_PERIGEO + 4.067e8) / 2)))

    # que unidad???
    MULTIPLICADOR = 1083
    x0  = -52409.924647197156 * MULTIPLICADOR
    y0  = -48671.635720228245 * MULTIPLICADOR
    vx0 = -1.22735334971968  * MULTIPLICADOR
    vy0 = -2.34643943970753  * MULTIPLICADOR

    estado_inicial = np.array([x0, y0, vx0, vy0])
    print(f"[comparar_artemis] Estado inicial: x={x0:.3e} y={y0:.3e} vx={vx0:.3e} vy={vy0:.3e}")
    print(f"[comparar_artemis] Fuentes: {args.metodos}")

    fuentes = resolver_fuentes(
        args.metodos, args.dt, args.duracion,
        estado_inicial,
        exportar_csv=args.exportar_csv,
    )
    graficar(fuentes, salida=args.salida, dpi=args.dpi)


if __name__ == '__main__':
    main()