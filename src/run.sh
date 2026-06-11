#!/usr/bin/env bash
# run.sh — Ejecuta scripts de simulación desde src/
# Uso:
#   ./run.sh luna --metodos rk4
#   ./run.sh luna --metodos rk4 euler --dt 60 --duracion 27.3
#   ./run.sh orion --metodo rk4 --duracion 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$1" in
    luna)
        shift
        PYTHONPATH="$SCRIPT_DIR" python graficos/trayectoria_luna.py "$@"
        ;;
    orion)
        shift
        PYTHONPATH="$SCRIPT_DIR" python graficos/trayectoria_orion.py "$@"
        ;;
    *)
        echo "Uso: $0 {luna|orion} [parámetros...]"
        echo ""
        echo "Ejemplos:"
        echo "  $0 luna --metodos rk4"
        echo "  $0 luna --metodos rk4 euler --dt 60 --duracion 27.3"
        echo "  $0 luna --metodos rk4 euler --dt 30 --duracion 27.3 --salida comparacion.png"
        echo "  $0 orion --metodo rk4"
        echo "  $0 orion --metodo rk4 --dt 60 --duracion 10"
        exit 1
        ;;
esac