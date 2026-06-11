#!/usr/bin/env python3
"""
trayectoria_orion.py

Requiere que exista el CSV de órbita lunar correspondiente al método elegido
(generado previamente con trayectoria_luna.py).

Uso:
    python trayectoria_orion.py --metodo rk4
    python trayectoria_orion.py --metodo euler --dt 60 --duracion 10
"""

import argparse
import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from utils.discretizaciones import INTEGRADORES
from utils.fisica import simular_orion, ORION_X0, ORION_Y0, ORION_VX0, ORION_VY0
from graficos.io_datos import cargar_posiciones_luna
from graficos.graficos_orion import VentanaOrbital


def construir_calcular_fuentes(metodo, dt, duracion) -> callable:
    """Devuelve una función callable(ev0, ev1)"""

    ruta_luna = f"resultados/resultado_{metodo.lower()}.csv"
    if not os.path.exists(ruta_luna):
        print(
            f"Error: no se encontró '{ruta_luna}'.\n"
            f"Ejecutá primero: python trayectoria_luna.py --metodos {metodo}",
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
        description="Simulador interactivo de la trayectoria de Orión.",
        epilog=__doc__,
    )
    p.add_argument("--metodo",   required=True,
                   help="Método de integración: rk4, euler, rk2")
    p.add_argument("--dt",       type=float, default=60.0,
                   help="Paso de tiempo en segundos (default: 60)")
    p.add_argument("--duracion", type=float, default=10.0,
                   help="Duración en días (default: 10)")
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

    app     = QApplication(sys.argv)
    ventana = VentanaOrbital(calcular_fuentes)
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()