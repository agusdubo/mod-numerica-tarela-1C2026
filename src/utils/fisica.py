# Código hecho por: Agustín Dubovitsky Otero (padrón: 111954) y Tomás Bautista Conti (Padrón: 111760)
import numpy as np
import pandas as pd

from src.utils.pasos import INTEGRADORES

# ── Constantes ───────────────────────────────────────────────────────────────
G               = 6.674e-11
M_TIERRA        = 5.972e24
M_LUNA          = 7.342e22
RADIO_TIERRA_KM = 6371
RADIO_LUNA_KM   = 1737

# Órbita lunar (perigeo/apogeo reales)
R_PERIGEO  = 3.565e8
R_APOGEO   = 4.067e8
A_SEMIEJE  = (R_PERIGEO + R_APOGEO) / 2
V_PERIGEO  = np.sqrt(G * M_TIERRA * (2 / R_PERIGEO - 1 / A_SEMIEJE))

# Condiciones iniciales de Orión (posición en m, velocidad en m/s)
ORION_X0  = -52409.924647197156 * 1_000
ORION_Y0  = -48671.635720228245 * 1_000
ORION_VX0 = -1.22735334971968   * 1_000
ORION_VY0 = -2.34643943970753   * 1_000


# ── Dinámica Luna (solo Tierra) ─────────────────────────────
def derivadas_luna(t, estado):
    x, y, vx, vy = estado
    d  = np.hypot(x, y)
    a  = -G * M_TIERRA / d**2
    return np.array([vx, vy, a * x / d, a * y / d])


# ── Dinámica Orión (Tierra + Luna) ────────────────────────────────────────
def _aceleracion_orion(x, y, pos_luna):
    # Tierra
    r_t = np.hypot(x, y)
    a_t = -G * M_TIERRA / r_t**2
    ax  = a_t * x / r_t
    ay  = a_t * y / r_t

    # Luna
    dx, dy = x - pos_luna[0], y - pos_luna[1]
    r_l    = np.hypot(dx, dy)
    a_l    = -G * M_LUNA / r_l**2
    ax    += a_l * dx / r_l
    ay    += a_l * dy / r_l

    return ax, ay


def derivadas_orion(estado, pos_luna):
    x, y, vx, vy = estado
    ax, ay = _aceleracion_orion(x, y, pos_luna)
    return np.array([vx, vy, ax, ay])


# ── Simulaciones ─────────────────────────────────────────────────────────────
def simular_luna(metodo: str, dt: float, duracion_dias: float) -> pd.DataFrame:
    integrador = INTEGRADORES[metodo]
    pasos      = int(duracion_dias * 86_400 / dt)
    estado     = np.array([-R_PERIGEO, 0.0, 0.0, -V_PERIGEO])
    t          = 0.0

    registros = np.empty((pasos + 1, 5))
    registros[0] = [t, *estado]

    print(f"  [{metodo.upper()}] dt={dt}s | {duracion_dias} días | {pasos:,} pasos...", end=" ")
    for i in range(pasos):
        estado = integrador(derivadas_luna, t, estado, dt)
        t     += dt
        registros[i + 1] = [t, *estado]

    print(f"listo ({pasos + 1:,} filas)")
    return pd.DataFrame(registros, columns=["time_s", "x", "y", "vx", "vy"])


def simular_orion(metodo: str, dt: float, duracion_dias: float, estado_inicial: np.ndarray, posiciones_luna: np.ndarray) -> pd.DataFrame:
    integrador = INTEGRADORES[metodo]
    pasos      = int(duracion_dias * 86_400 / dt)
    n_luna     = len(posiciones_luna)
    estado     = estado_inicial.copy()
    t          = 0.0

    registros = np.empty((pasos + 1, 5))
    registros[0] = [t, *estado]

    for i in range(pasos):
        pos_luna = posiciones_luna[min(i, n_luna - 1)]
        f        = lambda t, s: derivadas_orion(s, pos_luna)
        estado   = integrador(f, t, estado, dt)
        t       += dt
        registros[i + 1] = [t, *estado]

    return pd.DataFrame(registros, columns=["time_s", "x", "y", "vx", "vy"])