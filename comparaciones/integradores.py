"""
Integradores numéricos para EDOs de la forma:
    d(estado)/dt = f(t, estado)

Cada función recibe:
    f      : callable(t, estado) -> np.ndarray
    t      : float, tiempo actual
    estado : np.ndarray, estado actual
    h      : float, paso de tiempo

Retorna el estado en t + h.
"""

import numpy as np


def paso_euler(f, t, estado, h):
    """Euler explícito — orden 1."""
    return estado + h * f(t, estado)


def paso_rk4(f, t, estado, h):
    """Runge-Kutta clásico — orden 4."""
    k1 = f(t,       estado)
    k2 = f(t + h/2, estado + h/2 * k1)
    k3 = f(t + h/2, estado + h/2 * k2)
    k4 = f(t + h,   estado + h   * k3)
    return estado + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def paso_rk2(f, t, estado, h):
    k1 = f(t, estado)
    k2 = f(t + h/2,estado + h/2 * k1)
    return estado + h * k2


# Registro: nombre → función
INTEGRADORES = {
    'euler': paso_euler,
    'rk2':   paso_rk2,
    'rk4':   paso_rk4,
}