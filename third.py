from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def euler_method(
    f: Callable[[float, float], float], x0: float, y0: float, h: float, x_end: float
) -> Tuple[FloatArray, FloatArray]:
    x_values: List[float] = [x0]
    y_values: List[float] = [y0]

    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]

        y_new = y + h * f(x, y)
        x_new = x + h

        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(x_values, dtype=np.float64), np.array(y_values, dtype=np.float64)


def improved_euler_method(
    f: Callable[[float, float], float], x0: float, y0: float, h: float, x_end: float
) -> Tuple[FloatArray, FloatArray]:
    x_values: List[float] = [x0]
    y_values: List[float] = [y0]

    while x_values[-1] < x_end:
        x = x_values[-1]
        y = y_values[-1]

        y_p = y + h * f(x, y)

        y_new = y + h * 0.5 * (f(x, y) + f(x + h, y_p))
        x_new = x + h

        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(x_values, dtype=np.float64), np.array(y_values, dtype=np.float64)


def rk4_method(
    f: Callable[[float, float], float], x0: float, y0: float, h: float, x_end: float
) -> Tuple[FloatArray, FloatArray]:
    x0_float = np.float64(x0)
    y0_float = np.float64(y0)
    h_float = np.float64(h)
    x_end_float = np.float64(x_end)

    x_values = [x0_float]
    y_values = [y0_float]

    tol = np.finfo(float).eps * 10

    while x_values[-1] + tol < x_end_float:
        x = x_values[-1]
        y = y_values[-1]

        if x + h_float > x_end_float:
            h_float = x_end_float - x

        k1 = h_float * f(x, y)
        k2 = h_float * f(x + 0.5 * h_float, y + 0.5 * k1)
        k3 = h_float * f(x + 0.5 * h_float, y + 0.5 * k2)
        k4 = h_float * f(x + h_float, y + k3)

        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_new = x + h_float

        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(x_values, dtype=np.float64), np.array(y_values, dtype=np.float64)


def plot_methods(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
    exact_solution: Optional[Callable[[float | FloatArray], float | FloatArray]] = None,
) -> None:
    x_euler, y_euler = euler_method(f, x0, y0, h, x_end)
    x_ie, y_ie = improved_euler_method(f, x0, y0, h, x_end)
    x_rk4, y_rk4 = rk4_method(f, x0, y0, h, x_end)

    fig, ax = plt.subplots(figsize=(10, 6))

    if exact_solution is not None:
        y_exact = exact_solution(x_euler)  # TODO: handle different x values for exact solution
        ax.plot(x_euler, y_exact, "k-", label="Exact Solution")

    ax.plot(x_euler, y_euler, "b-.", label="Euler Method")
    ax.plot(x_ie, y_ie, "g-.", label="Improved Euler")
    ax.plot(x_rk4, y_rk4, "r-.", label="RK4")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(prop={"size": 12}).get_frame().set_alpha(None)
    plt.grid(True)

    axins = zoomed_inset_axes(ax, zoom=4, loc="lower left")

    if exact_solution is not None:
        axins.plot(x_euler, y_exact, "k-")
    axins.plot(x_euler, y_euler, "b-.")
    axins.plot(x_ie, y_ie, "g-.")
    axins.plot(x_rk4, y_rk4, "r-.")

    axins.set_xlim(1.75, 2.25)
    axins.set_ylim(0.6, 1.15)

    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.savefig("third_methods.png", bbox_inches="tight", dpi=300)


def plot_errors(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
    exact_solution: Callable[[float | FloatArray], float | FloatArray],
) -> None:
    x_euler, y_euler = euler_method(f, x0, y0, h, x_end)
    x_ie, y_ie = improved_euler_method(f, x0, y0, h, x_end)
    x_rk4, y_rk4 = rk4_method(f, x0, y0, h, x_end)

    y_exact_euler = exact_solution(x_euler)
    y_exact_ie = exact_solution(x_ie)
    y_exact_rk4 = exact_solution(x_rk4)

    error_euler = np.abs(y_euler - y_exact_euler)
    error_ie = np.abs(y_ie - y_exact_ie)
    error_rk4 = np.abs(y_rk4 - y_exact_rk4)

    plt.figure(figsize=(10, 6))
    plt.semilogy(x_euler, error_euler, "b--", label="Euler Method")
    plt.semilogy(x_ie, error_ie, "g--", label="Improved Euler")
    plt.semilogy(x_rk4, error_rk4, "r--", label="RK4")

    plt.xlabel("$x$")
    plt.ylabel("Error")
    plt.legend(prop={"size": 12}).get_frame().set_alpha(None)
    plt.grid(True)

    plt.savefig("third_errors.png", bbox_inches="tight", dpi=300)


def plot_step_size_errors(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    exact_solution: Callable[[float | FloatArray], float | FloatArray],
    step_sizes: List[float],
) -> None:
    euler_errors = []
    improved_euler_errors = []
    rk4_errors = []

    for h in step_sizes:
        x_euler, y_euler = euler_method(f, x0, y0, h, x_end)
        x_ie, y_ie = improved_euler_method(f, x0, y0, h, x_end)
        x_rk4, y_rk4 = rk4_method(f, x0, y0, h, x_end)

        y_exact_euler = exact_solution(x_euler)
        y_exact_ie = exact_solution(x_ie)
        y_exact_rk4 = exact_solution(x_rk4)

        euler_error = np.mean(np.abs(y_euler - y_exact_euler))
        ie_error = np.mean(np.abs(y_ie - y_exact_ie))
        rk4_error = np.mean(np.abs(y_rk4 - y_exact_rk4))

        euler_errors.append(euler_error)
        improved_euler_errors.append(ie_error)
        rk4_errors.append(rk4_error)

    step_sizes_array = np.array(step_sizes)
    euler_errors_array = np.array(euler_errors)
    improved_euler_errors_array = np.array(improved_euler_errors)
    rk4_errors_array = np.array(rk4_errors)

    plt.figure(figsize=(10, 6))

    plt.loglog(step_sizes, euler_errors, "bo-", label="Euler Method")

    h_ref = step_sizes_array.copy()
    euler_start = euler_errors_array[0]
    h_start = step_sizes_array[0]
    h_ref_values = euler_start * (h_ref / h_start)
    plt.loglog(h_ref, h_ref_values, "b:", alpha=0.7, label="$O(h)$")

    plt.loglog(step_sizes, improved_euler_errors, "gs-", label="Improved Euler")

    h2_ref = step_sizes_array.copy()
    ie_start = improved_euler_errors_array[0]
    h2_start = step_sizes_array[0]
    h2_ref_values = ie_start * (h2_ref / h2_start) ** 2
    plt.loglog(h2_ref, h2_ref_values, "g:", alpha=0.7, label="$O(h^2)$")

    plt.loglog(step_sizes, rk4_errors, "r^-", label="RK4")

    h4_ref = step_sizes_array.copy()
    rk4_start = rk4_errors_array[0]
    h4_start = step_sizes_array[0]
    h4_ref_values = rk4_start * (h4_ref / h4_start) ** 4
    plt.loglog(h4_ref, h4_ref_values, "r:", alpha=0.7, label="$O(h^4)$")

    plt.xlabel("Step Size, $h$")
    plt.ylabel("Average Error")
    plt.legend(prop={"size": 12}).get_frame().set_alpha(None)
    plt.grid(True)

    plt.savefig("third_step_size_errors.png", bbox_inches="tight", dpi=300)


def print_errors(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    exact_solution: Callable[[float | FloatArray], float | FloatArray],
    step_sizes: List[float],
) -> None:
    for h in step_sizes:
        x_euler, y_euler = euler_method(f, x0, y0, h, x_end)
        x_ie, y_ie = improved_euler_method(f, x0, y0, h, x_end)
        x_rk4, y_rk4 = rk4_method(f, x0, y0, h, x_end)

        y_exact_euler = exact_solution(x_euler)
        y_exact_ie = exact_solution(x_ie)
        y_exact_rk4 = exact_solution(x_rk4)

        error_euler = np.abs(y_euler - y_exact_euler)
        error_ie = np.abs(y_ie - y_exact_ie)
        error_rk4 = np.abs(y_rk4 - y_exact_rk4)

        print(f"N: {(x_end - x0) / h:.0f}")
        print("Euler Method Error:")
        print(error_euler)
        print("{:.2e}".format(np.mean(error_euler)))
        print("Improved Euler Method Error:")
        print(error_ie)
        print("{:.2e}".format(np.mean(error_ie)))
        print("RK4 Method Error:")
        print(error_rk4)
        print("{:.2e}".format(np.mean(error_rk4)))
        print()


if __name__ == "__main__":

    def f_xy(x: float, y: float) -> float:
        """
        y' = 5sin(x) + 4cos(2x) - 2y
        """
        return 5 * np.sin(x) + 4 * np.cos(2 * x) - 2 * y  # type: ignore

    def f_x(x: float | FloatArray) -> float | FloatArray:
        """
        y = 2sin(x) - cos(x) + cos(2x) + sin(2x)
        """
        return 2 * np.sin(x) - np.cos(x) + np.cos(2 * x) + np.sin(2 * x)  # type: ignore

    x0 = 0.0
    y0 = 0.0
    h = 0.15
    x_end = 5

    plot_methods(f_xy, x0, y0, h, x_end, f_x)

    plot_errors(f_xy, x0, y0, h, x_end, f_x)

    step_sizes = [0.001, 0.01, 0.1, 1.0]
    plot_step_size_errors(f_xy, x0, y0, x_end, f_x, step_sizes)

    N = [10, 20, 40, 80]
    step_sizes = [(x_end - x0) / n for n in N]
    print_errors(f_xy, x0, y0, x_end, f_x, step_sizes)

    plt.show()
