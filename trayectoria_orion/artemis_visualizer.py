import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Cargar los archivos CSV
# Reemplaza 'cuerpo1.csv' y 'cuerpo2.csv' con las rutas de tus archivos
try:
    df1 = pd.read_csv('resultado_rk4.csv')
    df2 = pd.read_csv('trayectoria_rk4.csv')
except FileNotFoundError as e:
    print(f"Error: Asegúrate de que los archivos CSV existan. {e}")
    sys.exit(1)

# 2. Determinar el rango de tiempo global (unificado)
# Buscamos el mínimo y máximo tiempo común para que el slider cubra toda la simulación
t_min = min(df1['time_s'].min(), df2['time_s'].min())
t_max = max(df1['time_s'].max(), df2['time_s'].max())

# 3. Configurar la interfaz gráfica (Matplotlib con soporte Qt)
# Forzamos el uso de Qt de manera explícita si está disponible
try:
    import matplotlib
    matplotlib.use('Qt5Agg') 
except ImportError:
    pass # Si falla, usa el backend por defecto del sistema

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2) # Dejamos espacio abajo para el slider

# Graficar las órbitas completas como líneas de fondo
ax.plot(df1['x'], df1['y'], color='blue', alpha=0.3, label='Órbita Cuerpo 1')
ax.plot(df2['x'], df2['y'], color='red', alpha=0.3, label='Órbita Cuerpo 2')

# Crear los elementos visuales interactivos (los puntos que se van a mover)
punto_cuerpo1, = ax.plot([], [], 'bo', markersize=8, label='Cuerpo 1')
punto_cuerpo2, = ax.plot([], [], 'ro', markersize=8, label='Cuerpo 2')

# Configuración estética del gráfico
ax.set_title("Visualizador Interactivo de Órbitas 2D")
ax.set_xlabel("Posición X")
ax.set_ylabel("Posición Y")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
ax.set_aspect('equal', 'datalim') # Mantiene la proporción escala 1:1 en los ejes

# 4. Crear el Slider de Tiempo
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03]) # Posición [izquierda, abajo, ancho, alto]
slider_tiempo = Slider(
    ax=ax_slider,
    label='Tiempo (s)',
    valmin=t_min,
    valmax=t_max,
    valinit=t_min,
    valfmt='%0.1f s'
)

# 5. Función de actualización (Interpolación)
def actualizar(val):
    t_actual = slider_tiempo.val
    
    # Interpolamos X e Y para el Cuerpo 1 en el tiempo actual
    x1 = np.interp(t_actual, df1['time_s'], df1['x'])
    y1 = np.interp(t_actual, df1['time_s'], df1['y'])
    
    # Interpolamos X e Y para el Cuerpo 2 en el tiempo actual
    x2 = np.interp(t_actual, df2['time_s'], df2['x'])
    y2 = np.interp(t_actual, df2['time_s'], df2['y'])
    
    # Actualizamos la posición de los puntos en el gráfico
    punto_cuerpo1.set_data([x1], [y1])
    punto_cuerpo2.set_data([x2], [y2])
    
    # Redibujar la figura de forma eficiente
    fig.canvas.draw_idle()

# Conectar el slider a la función de actualización
slider_tiempo.on_changed(actualizar)

# Inicializar la posición de los puntos en el tiempo t_min
actualizar(t_min)

plt.show()