#!/usr/bin/env python3
"""
Código hecho por: Agustín Dubovitsky Otero (padrón: 111954) y Tomás Bautista Conti (Padrón: 111760)
Uso:
    python trayectoria_luna.py --metodos rk4
    python trayectoria_luna.py --metodos rk4 euler rk2
    python trayectoria_luna.py --metodos rk4 euler --dt 60 --duracion 27.3
    python trayectoria_luna.py --metodos rk4 euler --salida comparacion.png

    dt       : paso de tiempo en segundos (default: 60)
    duracion : duración de la simulación en días (default: 27.3)
"""

import argparse
import os
import sys

from src.utils.pasos import INTEGRADORES
from utils.fisica import simular_luna
from graficos.io_datos  import cargar_csv_trayectoria, guardar_csv
from graficos.graficos_luna import graficar_orbitas


def resolver_fuentes(metodos, dt, duracion, exportar_csv):
    """Devuelve lista de (etiqueta, DataFrame) para cada método/CSV."""
    fuentes = []
    for m in metodos:
        if m.startswith("csv:"):
            ruta = m[4:]
            if not os.path.exists(ruta):
                print(f"Error: no se encontró '{ruta}'", file=sys.stderr)
                sys.exit(1)
            df  = cargar_csv_trayectoria(ruta)
            etq = os.path.splitext(os.path.basename(ruta))[0]

        elif m in INTEGRADORES:
            df  = simular_luna(m, dt, duracion)
            etq = m.upper()
            if exportar_csv:
                guardar_csv(df, f"resultados/resultado_{m.lower()}.csv")

        else:
            print(f"Error: método desconocido '{m}'. Opciones: rk4, euler, rk2, csv:<ruta>",
                  file=sys.stderr)
            sys.exit(1)

        fuentes.append((etq, df))
    return fuentes


def parse_args():
    p = argparse.ArgumentParser(
        description="Compara trayectorias orbitales lunares.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--metodos", nargs="+", required=True, metavar="MÉTODO",
                   help="Uno o más métodos: rk4, euler, rk2, csv:<ruta>")
    p.add_argument("--dt",       type=float, default=60.0,
                   help="Paso de tiempo en segundos (default: 60)")
    p.add_argument("--duracion", type=float, default=27.3,
                   help="Duración en días (default: 27.3)")
    p.add_argument("--salida",   type=str,
                   default="resultados/comparacion_orbital.png",
                   help="Ruta de la imagen de salida")
    p.add_argument("--dpi",      type=int, default=150,
                   help="Resolución de la imagen (default: 150)")
    p.add_argument("--no-exportar-csv", dest="exportar_csv",
                   action="store_false",
                   help="Desactiva la exportación de CSV (activada por defecto)")
    return p.parse_args()


def main():
    args    = parse_args()
    fuentes = resolver_fuentes(args.metodos, args.dt, args.duracion,
                               args.exportar_csv)
    graficar_orbitas(fuentes, salida=args.salida, dpi=args.dpi)


if __name__ == "__main__":
    main()