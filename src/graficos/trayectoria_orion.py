#!/usr/bin/env python3
"""
Código hecho por: Agustín Dubovitsky Otero (padrón: 111954) y Tomás Bautista Conti (Padrón: 111760)
Requiere que exista el CSV de órbita lunar correspondiente al método elegido
(generado previamente con trayectoria_luna.py).

Uso:
    python trayectoria_orion.py --metodo rk4
    python trayectoria_orion.py --metodo euler --dt 60 --duracion 10
    python trayectoria_orion.py --metodo rk4 --ev0 -633 --ev1 -36 --paso 100 --salida mi_imagen.png
"""

import argparse
import os
import sys

import numpy as np

from utils.pasos import INTEGRADORES
from utils.fisica import simular_orion, ORION_X0, ORION_Y0, ORION_VX0, ORION_VY0
from graficos.io_datos import cargar_posiciones_luna
from graficos.graficos_orion import graficar_orbital


def construir_calcular_fuentes(metodo, dt, duracion) -> callable:
    """Devuelve una función callable(ev0, ev1)"""

    ruta_luna = f"resultados/resultado_{metodo.lower()}.csv"
    if not os.path.exists(ruta_luna):
        print(
            f"Error: no se encontró '{ruta_luna}'.\n"
            f"Ejecutá primero trayectoria_luna.py --metodos {metodo}",
            file=sys.stderr,
        )
        sys.exit(1)

    posiciones_luna, df_luna = cargar_posiciones_luna(ruta_luna)

    def calcular_fuentes(ev0: float, ev1: float):
        estado_inicial = np.array([
            ORION_X0, ORION_Y0,
            ORION_VX0 + ev0,
            ORION_VY0 + ev1,
        ])
        df_orion = simular_orion(metodo, dt, duracion, estado_inicial,
                                 posiciones_luna)
        return [(metodo.upper(), df_orion, df_luna)]

    return calcular_fuentes


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulador de la trayectoria de Orión (genera imagen).",
        epilog=__doc__,
    )
    p.add_argument("--metodo",   required=True,
                   help="Método de integración: rk4, euler, rk2")
    p.add_argument("--dt",       type=float, default=60.0,
                   help="Paso de tiempo en segundos (default: 60)")
    p.add_argument("--duracion", type=float, default=10.0,
                   help="Duración en días (default: 10)")
    p.add_argument("--ev0",      type=float, default=-633.0,
                   help="extra_v[0] velocidad inicial en x (default: -633)")
    p.add_argument("--ev1",      type=float, default=-36.0,
                   help="extra_v[1] velocidad inicial en y (default: -36)")
    p.add_argument("--paso",     type=int,   default=0,
                   help="Paso de tiempo a marcar en la imagen (default: 0)")
    p.add_argument("--salida",   type=str,   default="simulacion_orbital.png",
                   help="Nombre del archivo de imagen a guardar (default: simulacion_orbital.png)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.metodo not in INTEGRADORES:
        print(f"Error: método '{args.metodo}' no reconocido. "
              f"Opciones: {', '.join(INTEGRADORES)}",
              file=sys.stderr)
        sys.exit(1)

    calcular_fuentes = construir_calcular_fuentes(
        args.metodo, args.dt, args.duracion
    )

    graficar_orbital(
        calcular_fuentes,
        ev0=args.ev0,
        ev1=args.ev1,
        paso=args.paso,
        salida=args.salida,
    )


if __name__ == "__main__":
    main()