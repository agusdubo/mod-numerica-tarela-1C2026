import numpy as np
import sys

M_T = 5.972e24  # Masa de la Tierra [kg]
M_L = 7.348e22  # Masa de la Luna [kg]

CONST_G = 6.67e-11

T_TOTAL = 27.3 * 24 * 3600   # Duración total = 1 período orbital ≈ 2 359 440 s
DT      = 60                 # Paso de tiempo = 60 segundos (sincronizado con datos de Luna)

# Cargar trayectoria de la Luna
ruta_luna = sys.argv[1] if len(sys.argv) > 1 else '../luna/trayectoria_euler_luna.csv'
datos_luna = np.loadtxt(ruta_luna, delimiter=',', skiprows=1, usecols=(1, 2))
tiempos_luna = np.loadtxt(ruta_luna, delimiter=',', skiprows=1, usecols=0)

# ============================================================================
# VALORES INICIALES DE LA NAVE - EDITAR AQUÍ
# ============================================================================
# Datos de telemetría Artemis II (convertidos de km y km/s a m y m/s)
# Original: 2026-04-03T05:03:39,017;-52409,924647197156;-48671,635720228245;-27311,232327946513;-1,22735334971968;-2,34643943970753;-1,29054721385023
x0  = -52409.924647197156 * 1e3   # m  (de km a m, perigeo lunar)
y0  = -48671.635720228245 * 1e3        # m  (de km a m)
vx0 = -1.22735334971968 * 1e3        # m/s  (de km/s a m/s)
vy0 = -2.34643943970753 * 1e3          # m/s  (de km/s a m/s, velocidad orbital)
# ============================================================================

def euler_step(state, dt, acceleration_func, pos_luna):
    """
    Realiza un paso de integración usando el método de Euler.

    Parámetros:
    - state: tupla (x, y, vx, vy) representando la posición y velocidad actuales.
    - dt: paso de tiempo a avanzar.
    - acceleration_func: función que toma (x, y, pos_luna) y devuelve (ax, ay).
    - pos_luna: tupla (x_luna, y_luna) con la posición de la Luna.

    Retorna:
    - new_state: tupla (new_x, new_y, new_vx, new_vy) con la nueva posición y velocidad.
    """
    x, y, vx, vy = state
    ax, ay = acceleration_func(x, y, pos_luna)

    # Actualizar velocidad
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt

    # Actualizar posición
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    return (new_x, new_y, new_vx, new_vy)

def gravitational_acceleration(x, y, pos_luna):
    """
    Calcula la aceleración gravitacional total como suma vectorial
    de las aceleraciones debidas a la Tierra y la Luna.

    Parámetros:
    - x, y: coordenadas del punto donde se calcula la aceleración.
    - pos_luna: tupla (x_luna, y_luna) con la posición de la Luna.

    Retorna:
    - ax, ay: componentes de la aceleración gravitacional total.
    """
    # Aceleración debida a la Tierra
    r_tierra_squared = x**2 + y**2
    r_tierra = r_tierra_squared**0.5
    if r_tierra == 0:
        ax_tierra = 0
        ay_tierra = 0
    else:
        a_mag_tierra = CONST_G * M_T / r_tierra_squared
        ax_tierra = -a_mag_tierra * (x / r_tierra)
        ay_tierra = -a_mag_tierra * (y / r_tierra)
    
    # Aceleración debida a la Luna
    r_luna_squared = (x - pos_luna[0])**2 + (y - pos_luna[1])**2
    r_luna = r_luna_squared**0.5
    if r_luna == 0:
        ax_luna = 0
        ay_luna = 0
    else:
        a_mag_luna = CONST_G * M_L / r_luna_squared
        ax_luna = -a_mag_luna * ((x - pos_luna[0]) / r_luna)
        ay_luna = -a_mag_luna * ((y - pos_luna[1]) / r_luna)
    
    # Suma vectorial
    ax = ax_tierra + ax_luna
    ay = ay_tierra + ay_luna

    return (ax, ay)

estado = np.array([x0, y0, vx0, vy0])
t      = 0.0
pasos  = int(T_TOTAL / DT)   # Calcular pasos basado en T_TOTAL y DT

tiempos     = np.zeros(pasos + 1)
trayectoria = np.zeros((pasos + 1, 4))

# Guardar estado inicial
trayectoria[0] = estado
tiempos[0]     = t

for i in range(pasos):
    # Posición actual de la Luna desde los datos cargados
    pos_luna = datos_luna[i, :2]  # Solo X, Y
    estado = euler_step(estado, DT, gravitational_acceleration, pos_luna)
    t     += DT
    trayectoria[i+1] = estado
    tiempos[i+1]     = t

# Guardar en CSV
datos = np.column_stack([tiempos, trayectoria])
np.savetxt('trayectoria_euler_artemis.csv', datos, delimiter=',', header='tiempo,x,y,vx,vy', comments='')


