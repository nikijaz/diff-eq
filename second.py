import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

G = 0.1
K = 250
P0 = [2400, 2500, 2600]

t = np.linspace(0, 50, 20)
P = np.linspace(0, 5000, 20)
T, P_grid = np.meshgrid(np.linspace(0, 50, 20), np.linspace(0, 5000, 20))

dPdt = G * P_grid - K

U = np.ones_like(dPdt)
V = dPdt
N = np.sqrt(U**2 + V**2)

U_norm = U / N
V_norm = V / N

norm = Normalize()
norm.autoscale(N)

colormap = plt.colormaps.get_cmap("coolwarm")

plt.figure(figsize=(10, 6))

quiv = plt.quiver(T, P_grid, U_norm, V_norm, N, angles="xy", pivot="mid", alpha=0.15, cmap=colormap, norm=norm)

values = [">", "=", "<"]
for i, P0_val in enumerate(reversed(P0)):
    P_solution = K / G + (P0_val - K / G) * np.exp(G * t)
    plt.plot(t, P_solution, linewidth=2.5, label=f"$P_0 = {P0_val} {values[i]} \\frac{{H}}{{r}}$")

plt.xlabel("Time (years), $t$")
plt.ylabel("Population, $P$")
plt.legend(prop={"size": 12}).get_frame().set_alpha(None)
plt.grid(True)
plt.ylim(0, 5000)
plt.xlim(0, 50)
plt.savefig("first_plot.png", bbox_inches="tight", dpi=300)
