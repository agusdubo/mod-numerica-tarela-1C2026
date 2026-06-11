def paso_euler(f, t, estado, h):
    return estado + h * f(t, estado)


def paso_rk4(f, t, estado, h):
    q1 = f(t,       estado)
    q2 = f(t + h/2, estado + h/2 * q1)
    q3 = f(t + h/2, estado + h/2 * q2)
    q4 = f(t + h,   estado + h   * q3)
    return estado + (h / 6) * (q1 + 2*q2 + 2*q3 + q4)


def paso_rk2(f, t, estado, h):
    q1 = f(t, estado)
    q2 = f(t + h/2,estado + h/2 * q1)
    return estado + h * q2


INTEGRADORES = {
    'euler': paso_euler,
    'rk2':   paso_rk2,
    'rk4':   paso_rk4,
}