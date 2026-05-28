"""
Simulación de la trayectoria de la Luna usando Runge-Kutta 4
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ── Constantes físicas ──────────────────────────────────────────────────────
G   = 6.674e-11
M_T = 5.972e24

# ── Condiciones iniciales ────────────────────────────────────────────────────
x0  =  3.633e8
y0  =  0.0
vx0 =  0.0
vy0 =  1.082e3

# ── Parámetros de integración ────────────────────────────────────────────────
T_TOTAL = 27.3 * 24 * 3600
DT      = 60.0

# ── Sistema de EDOs ──────────────────────────────────────────────────────────
def derivadas(t, estado):
    x, y, vx, vy = estado
    d2    = x**2 + y**2
    d     = np.sqrt(d2)
    ax = -G * M_T / d2 * (x / d)
    ay = -G * M_T / d2 * (y / d)
    return np.array([vx, vy, ax, ay])

# ── Runge-Kutta 4 ────────────────────────────────────────────────────────────
def rk4_paso(f, t, estado, h):
    k1 = f(t,         estado)
    k2 = f(t + h/2,   estado + h/2 * k1)
    k3 = f(t + h/2,   estado + h/2 * k2)
    k4 = f(t + h,     estado + h   * k3)
    return estado + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# ── Integración ──────────────────────────────────────────────────────────────
estado = np.array([x0, y0, vx0, vy0])
t      = 0.0
pasos  = int(T_TOTAL / DT)

trayectoria = np.zeros((pasos + 1, 4))
tiempos     = np.zeros(pasos + 1)
trayectoria[0] = estado

for i in range(pasos):
    estado = rk4_paso(derivadas, t, estado, DT)
    t     += DT
    trayectoria[i+1] = estado
    tiempos[i+1]     = t

x_tray = trayectoria[:, 0]
y_tray = trayectoria[:, 1]
dias   = tiempos / 86400

# ── Gráfico ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0e1a")
ax.set_facecolor("#050810")
ax.set_aspect("equal")

# Trayectoria con gradiente de tiempo
puntos   = np.array([x_tray, y_tray]).T.reshape(-1, 1, 2)
segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
lc = LineCollection(segmentos, cmap="cool", linewidth=1.4, alpha=0.9)
lc.set_array(dias[:-1])
ax.add_collection(lc)
cbar = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Tiempo (días)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

# Tierra
tierra = Circle((0, 0), 6.371e6 * 8, color="#1a6bb5", zorder=5)
ax.add_patch(tierra)
ax.text(0, -3.2e7, "Tierra", ha="center", color="#7ec8e3", fontsize=9)

# Luna (posición inicial)
luna = Circle((x0, y0), 1.737e6 * 8, color="#c8c8c8", zorder=5)
ax.add_patch(luna)
ax.text(x0, y0 + 3.5e7, "Luna (t=0)", ha="center", color="#ddd", fontsize=9)

ax.set_xlim(-4.2e8, 4.2e8)
ax.set_ylim(-4.2e8, 4.2e8)
ax.set_xlabel("x (m)", color="#aaa", fontsize=10)
ax.set_ylabel("y (m)", color="#aaa", fontsize=10)
ax.set_title("Trayectoria Orbital de la Luna — Runge-Kutta 4",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="#888", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")

def fmt(x, _):
    return f"{x/1e6:.0f}×10⁶"
ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))

plt.tight_layout()
plt.savefig("img/orbita_lunar_rk4.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
print("✓ Gráfico guardado")
plt.show()

# ── Segunda figura: muchas vueltas ───────────────────────────────────────────
N_ORBITAS_MULTI = 10
T_TOTAL_MULTI   = N_ORBITAS_MULTI * 27.3 * 24 * 3600
pasos_multi     = int(T_TOTAL_MULTI / DT)

estado_m       = np.array([x0, y0, vx0, vy0])
t_m            = 0.0
tray_multi     = np.zeros((pasos_multi + 1, 2))
tiempos_multi  = np.zeros(pasos_multi + 1)
tray_multi[0]  = [x0, y0]

for i in range(pasos_multi):
    estado_m          = rk4_paso(derivadas, t_m, estado_m, DT)
    t_m              += DT
    tray_multi[i+1]   = estado_m[:2]
    tiempos_multi[i+1] = t_m

dias_multi = tiempos_multi / 86400

fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor="#0a0e1a")
ax2.set_facecolor("#050810")
ax2.set_aspect("equal")

puntos_m    = tray_multi.reshape(-1, 1, 2)
segmentos_m = np.concatenate([puntos_m[:-1], puntos_m[1:]], axis=1)
lc2 = LineCollection(segmentos_m, cmap="plasma", linewidth=0.8, alpha=0.7)
lc2.set_array(dias_multi[:-1])
ax2.add_collection(lc2)
cbar2 = fig2.colorbar(lc2, ax=ax2, pad=0.02, fraction=0.03)
cbar2.set_label("Tiempo (días)", color="white", fontsize=9)
cbar2.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

tierra2 = Circle((0, 0), 6.371e6 * 8, color="#1a6bb5", zorder=5)
ax2.add_patch(tierra2)
ax2.text(0, -3.2e7, "Tierra", ha="center", color="#7ec8e3", fontsize=9)

ax2.set_xlim(-4.2e8, 4.2e8)
ax2.set_ylim(-4.2e8, 4.2e8)
ax2.set_xlabel("x (m)", color="#aaa", fontsize=10)
ax2.set_ylabel("y (m)", color="#aaa", fontsize=10)
ax2.set_title(f"Trayectoria Orbital de la Luna — {N_ORBITAS_MULTI} vueltas — Runge-Kutta 4",
              color="white", fontsize=13, pad=12)
ax2.tick_params(colors="#888", labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(fmt))

plt.tight_layout()
plt.savefig("img/orbita_lunar_rk4_multivuelta.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
print("✓ Gráfico multivuelta guardado")
plt.show()