import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def f_t(t: float | FloatArray) -> float | FloatArray:
    return T_r + (T_n - T_r) * np.exp(-k * t)


T_r = 20
T_n = 36.6
T_0 = 31
T_1 = 30
k = -math.log(10 / 11)
t_d = -1 / k * math.log((T_0 - T_r) / (T_n - T_r))

t = np.linspace(0, 10, 15, dtype=np.float64)
T = np.linspace(15, 40, 15, dtype=np.float64)
T_grid, t_grid = np.meshgrid(T, t)
dTdt = -k * (T_grid - T_r)

U = np.ones_like(dTdt)
V = dTdt
N = np.sqrt(U**2 + V**2)

U_norm = U / N
V_norm = V / N

norm = Normalize()
norm.autoscale(N)
colormap = plt.colormaps.get_cmap("coolwarm")


plt.figure(figsize=(10, 6))

plt.quiver(t_grid, T_grid, U_norm, V_norm, N, angles="xy", pivot="mid", alpha=0.15, cmap=colormap, norm=norm)

plt.plot(t, f_t(t), label="Body Temperature")
plt.axhline(y=T_n, color="g", linestyle="--", label="Normal Body Temperature ($36.6°C$)")
plt.axhline(y=T_r, color="r", linestyle="--", label="Room Temperature ($20°C$)")

plt.axvline(t_d, color="k", linestyle=":", label="Death Discovery Time")
plt.scatter(t_d, T_0, color="b", zorder=5, label=f"Discovered at ${T_0}°C$")
plt.scatter(t_d + 1, T_1, color="purple", zorder=5, label=f"One Hour Later at ${T_1}°C$")

plt.annotate(
    "Time of Death ($t = 0$)",
    xy=(0, T_n),
    xytext=(0.5, T_n + 1),
    arrowprops=dict(facecolor="black", width=1, headwidth=5, headlength=5),
    fontsize=10,
)

plt.xlabel("Time after death (hours), $t$")
plt.ylabel("Body Temperature (°$C$), $T$")
plt.grid(True)
plt.legend(prop={"size": 12}).get_frame().set_alpha(None)
plt.tight_layout()
plt.savefig("first_plot.png", bbox_inches="tight", dpi=300)
