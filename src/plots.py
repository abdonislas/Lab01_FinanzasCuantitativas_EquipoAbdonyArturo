"""
Generacion de las figuras del laboratorio.

Cada funcion recibe datos ya calculados (nunca los calcula) y devuelve la
figura de matplotlib, guardandola en disco si se indica una ruta.
Todas las figuras llevan titulo, ejes etiquetados y leyenda.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: permite ejecutar en cualquier maquina
import matplotlib.pyplot as plt
import numpy as np

from src.model import ModelParams, execution_probability

# Paleta unica para los tres regimenes, usada de forma consistente en todo el proyecto.
REGIME_COLORS = {
    "Optimo": "#1b6ca8",
    "Estrecho": "#c1440e",
    "Amplio": "#2e7d32",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_execution_probability(params: ModelParams,
                               path: str | Path | None = None) -> plt.Figure:
    """Figura 1: probabilidad de ejecucion contra el semi-spread.

    Marca explicitamente el punto donde la probabilidad llega a cero,
    s = alpha / beta = 6.25.
    """
    s_zero = params.zero_demand_spread
    s = np.linspace(0.0, s_zero * 1.4, 500)
    p = execution_probability(s, params)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(s, p, color="#1b6ca8", lw=2.2,
            label=r"$\pi_{LB}(s)=\pi_{LS}(s)=\max(0,\ 0.50-0.08\,s)$")
    ax.axvline(s_zero, color="#c1440e", ls="--", lw=1.4)
    ax.plot([s_zero], [0.0], "o", color="#c1440e", ms=9, zorder=5,
            label=f"Demanda nula: s = {s_zero:.2f}")
    ax.annotate(f"s = {s_zero:.2f}\n$\\pi_{{LB}}=0$",
                xy=(s_zero, 0.0), xytext=(s_zero + 0.5, 0.12),
                arrowprops=dict(arrowstyle="->", color="#c1440e"),
                color="#c1440e")

    ax.set_title("Probabilidad de ejecucion de un trader de liquidez\ncontra el semi-spread cotizado")
    ax.set_xlabel("Semi-spread $s$ respecto a $S_0$ (unidades monetarias)")
    ax.set_ylabel("Probabilidad de ejecucion")
    ax.set_ylim(-0.02, params.alpha * 1.15)
    ax.set_xlim(0, s_zero * 1.4)
    ax.legend(loc="upper right", frameon=True)
    return _save(fig, path)


def plot_cumulative_pnl(runs: dict[str, "object"],
                        path: str | Path | None = None) -> plt.Figure:
    """Figura 2: P&L acumulado a lo largo de los trades, tres regimenes."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name, df in runs.items():
        ax.plot(df["trade_id"], df["cum_pnl"], lw=1.6,
                color=REGIME_COLORS.get(name), label=name)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)

    n = len(next(iter(runs.values())))
    ax.set_title(f"P&L acumulado del formador de mercado ({n:,} trades)".replace(",", " "))
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("P&L acumulado (unidades monetarias)")
    ax.legend(title="Regimen de cotizacion", loc="upper left")
    return _save(fig, path)


def plot_inventory(runs: dict[str, "object"],
                   path: str | Path | None = None) -> plt.Figure:
    """Figura 3: inventario acumulado a lo largo de los trades, tres regimenes."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name, df in runs.items():
        ax.plot(df["trade_id"], df["inventory"], lw=1.4,
                color=REGIME_COLORS.get(name), label=name, alpha=0.9)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)

    n = len(next(iter(runs.values())))
    ax.set_title(f"Inventario acumulado del formador de mercado ({n:,} trades)".replace(",", " "))
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("Inventario neto (unidades del activo)")
    # Las trayectorias de inventario derivan hacia abajo: la esquina superior
    # derecha es la unica que queda libre en los tres regimenes.
    ax.legend(title="Regimen de cotizacion", loc="upper right")
    return _save(fig, path)


def plot_mc_histogram(mc_results: dict[str, np.ndarray],
                      path: str | Path | None = None) -> plt.Figure:
    """Figura 4: histograma del P&L final del analisis de Monte Carlo."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    all_values = np.concatenate(list(mc_results.values()))
    bins = np.linspace(all_values.min(), all_values.max(), 60)

    for name, values in mc_results.items():
        color = REGIME_COLORS.get(name)
        ax.hist(values, bins=bins, alpha=0.45, color=color,
                label=f"{name} (media = {values.mean():.1f})")
        ax.axvline(values.mean(), color=color, ls="--", lw=1.6)

    ax.axvline(0.0, color="black", lw=1.2, label="P&L = 0")
    n_runs = len(next(iter(mc_results.values())))
    ax.set_title(f"Distribucion del P&L final\n({n_runs:,} corridas de Monte Carlo)".replace(",", " "))
    ax.set_xlabel("P&L final de la corrida (unidades monetarias)")
    ax.set_ylabel("Frecuencia (numero de corridas)")
    ax.legend(title="Regimen de cotizacion", loc="upper left")
    return _save(fig, path)


def plot_spread_sensitivity(sensitivity_rows: list[dict],
                            params: ModelParams,
                            path: str | Path | None = None) -> plt.Figure:
    """Figura 5: spread optimo contra pi_I, con la referencia teorica pi_I = 0.

    La referencia horizontal es el spread del monopolista sin seleccion adversa,
    s* = alpha / beta = 6.25 (spread total), resultado analitico visto en la
    Sesion 04. Cualquier punto por encima de esa linea es el costo de la
    seleccion adversa traducido a spread.
    """
    pis = [row["pi_informed"] for row in sensitivity_rows]
    spreads = [row["spread"] for row in sensitivity_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(pis, spreads, "o-", color="#1b6ca8", lw=2.0, ms=8,
            label="Spread optimo numerico")
    ax.axhline(params.zero_demand_spread, color="#c1440e", ls="--", lw=1.6,
               label=(r"Referencia teorica $\pi_I=0$: "
                      f"$s^*=\\alpha/\\beta={params.zero_demand_spread:.2f}$"))
    for pi, sp in zip(pis, spreads):
        ax.annotate(f"{sp:.2f}", xy=(pi, sp), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)

    ax.set_title("Sensibilidad del spread optimo a la proporcion\nde traders informados")
    ax.set_xlabel(r"Probabilidad de trader informado $\pi_I$")
    ax.set_ylabel("Spread optimo total (Ask - Bid)")
    ax.legend(loc="upper left")
    return _save(fig, path)
