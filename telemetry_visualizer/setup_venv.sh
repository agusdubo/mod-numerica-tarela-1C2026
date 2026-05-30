#!/usr/bin/env bash
set -euo pipefail

# Crea un virtualenv local en .venv e instala dependencias
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Virtualenv creada en .venv"
echo "Actívala con: source .venv/bin/activate"
