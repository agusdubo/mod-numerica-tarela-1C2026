import numpy as np

M_T = 5.972e24  # Masa de la Tierra [kg]

CONST_G = 6.67e-11

T_TOTAL = 27.3 * 24 * 3600   # Duración total = 1 período orbital ≈ 2 359 440 s
DT      = 60.0                # Paso de tiempo = 60 segundos

r_perigeo = 3.565e8    # m  (356 500 km)
r_apogeo  = 4.067e8    # m  (406 700 km)
a         = (r_perigeo + r_apogeo) / 2   # semi-eje mayor
v_perigeo = np.sqrt(CONST_G * M_T * (2/r_perigeo - 1/a))
# Valores Iniciales
x0  = r_perigeo   # arranca en el perigeo
y0  = 0.0
vx0 = 0.0
vy0 = v_perigeo   # velocidad consistente con esa posición

def euler_step(state, dt, acceleration_func):
    """
    Realiza un paso de integración usando el método de Euler.

    Parámetros:
    - state: tupla (x, y, vx, vy) representando la posición y velocidad actuales.
    - dt: paso de tiempo a avanzar.
    - acceleration_func: función que toma (x, y) y devuelve (ax, ay).

    Retorna:
    - new_state: tupla (new_x, new_y, new_vx, new_vy) con la nueva posición y velocidad.
    """
    x, y, vx, vy = state
    ax, ay = acceleration_func(x, y)

    # Actualizar velocidad
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt

    # Actualizar posición
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    return (new_x, new_y, new_vx, new_vy)

def gravitational_acceleration(x, y):
    """
    Calcula la aceleración gravitacional en el punto (x, y) debido a una masa M en el origen.

    Parámetros:
    - x, y: coordenadas del punto donde se calcula la aceleración.
    - M: masa que genera el campo gravitacional.

    Retorna:
    - ax, ay: componentes de la aceleración gravitacional.
    """
    r_squared = x**2 + y**2
    r = r_squared**0.5
    if r == 0:
        return (0, 0)  # Evitar división por cero

    a_magnitude = CONST_G * M_T / r_squared
    ax = -a_magnitude * (x / r)
    ay = -a_magnitude * (y / r)

    return (ax, ay)

estado = np.array([x0, y0, vx0, vy0])
t      = 0.0
pasos  = int(T_TOTAL / DT)

tiempos     = np.zeros(pasos + 1)
trayectoria = np.zeros((pasos + 1, 4))

# Guardar estado inicial
trayectoria[0] = estado
tiempos[0]     = t

for i in range(pasos):
    estado = euler_step(estado, DT, gravitational_acceleration)
    t     += DT
    trayectoria[i+1] = estado
    tiempos[i+1]     = t

# Guardar en CSV
datos = np.column_stack([tiempos, trayectoria])
np.savetxt('trayectoria_euler.csv', datos, delimiter=',', header='tiempo,x,y,vx,vy', comments='')


