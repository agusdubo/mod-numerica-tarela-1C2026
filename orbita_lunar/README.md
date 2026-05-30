# Visualizador de órbita lunar

Unidad: kilometro

Setup y uso rápido:

1) Crear y activar virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Ejecutar (interactivo con slider)

```bash
python visualize_orbit.py "lunar_orbit.csv"
```

También hay `setup_venv.sh` para automatizar la creación del venv e instalación.
