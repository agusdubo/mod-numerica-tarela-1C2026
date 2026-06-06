#!/usr/bin/env python3
"""Script para arreglar el formato del CSV eliminando espacios extras alrededor de comas."""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Arregla el formato de archivos CSV eliminando espacios extras alrededor de comas.')
    parser.add_argument('file', help='Ruta del archivo CSV a arreglar')
    args = parser.parse_args()
    
    csv_file = args.file
    
    # Verificar que el archivo existe
    if not os.path.exists(csv_file):
        print(f"Error: El archivo '{csv_file}' no existe.", file=sys.stderr)
        sys.exit(1)
    
    # Leer el archivo
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # Arreglar cada línea eliminando espacios extras alrededor de comas
    fixed_lines = []
    for line in lines:
        # Dividir por comas y eliminar espacios alrededor de cada valor
        values = [val.strip() for val in line.split(',')]
        # Rejuntar sin espacios extras
        fixed_line = ','.join(values)
        fixed_lines.append(fixed_line + '\n')
    
    # Escribir el archivo arreglado
    with open(csv_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✓ Archivo {csv_file} arreglado correctamente.")
    print(f"  Se procesaron {len(fixed_lines)} líneas.")
    print(f"  Se eliminaron espacios extras alrededor de las comas.")

if __name__ == '__main__':
    main()
