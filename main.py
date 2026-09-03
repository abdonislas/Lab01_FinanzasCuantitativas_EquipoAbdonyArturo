from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# para poder importar src/ sin importar desde donde se corra el script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import plots
from src.model import (ModelParams, expected_informed_loss_ask,
                       expected_informed_loss_bid, expected_utility,
                       optimize_quotes, optimize_quotes_grid,
                       sensitivity_to_pi_informed)
from src.simulation import (DEFAULT_SEED, expected_inventory_drift,
                            monte_carlo_stats, run_monte_carlo,
                            simulate_trades, summarize_run,
                            validate_against_theory)

# Revisado hasta aqui (Arturo) - resto pendiente de revision

FIGURES_DIR = Path(__file__).resolve().parent / "docs" / "figures"

N_TRADES = 10_000
MC_RUNS = 1_000
MC_TRADES = 1_000
PI_GRID = (0.1, 0.4, 0.7)


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    # La consola de Windows usa cp1252 por defecto y rompe con caracteres no ASCII.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    np.random.seed(DEFAULT_SEED)  # semilla global documentada en el README
    params = ModelParams()

    # ------------------------------------------------------------------ 1 y 2
    _rule("1. PARAMETROS DEL CASO BASE")
    dist = params.value_dist
    print(f"  S0 (precio de referencia)     : {params.s0:.2f}")
    print(f"  P ~ Erlang(K={params.k}, lambda={params.lam:.0f})   : "
          f"E[P] = {dist.mean():.2f}, sd[P] = {dist.std():.2f}")
    print(f"  pi_I / pi_L                   : {params.pi_informed:.2f} / {params.pi_liquidity:.2f}")
    print(f"  Demanda no informada          : max(0, {params.alpha:.2f} - {params.beta:.2f}*s)")
    print(f"  Semi-spread de demanda nula   : {params.zero_demand_spread:.2f}")

    _rule("2. COTIZACIONES OPTIMAS (scipy.optimize.minimize)")
    sol = optimize_quotes(params)
    check = optimize_quotes_grid(params)
    print(f"  Bid optimo                    : {sol.bid:.2f}")
    print(f"  Ask optimo                    : {sol.ask:.2f}")
    print(f"  Spread optimo (Ask - Bid)     : {sol.spread:.2f}")
    print(f"  Utilidad esperada por trade   : {sol.expected_utility:.2f}")
    print(f"  Convergencia                  : {sol.converged} ({sol.message})")
    print(f"  Verificacion por barrido fino : Bid = {check.bid:.2f}, Ask = {check.ask:.2f}, "
          f"spread = {check.spread:.2f}")
    print(f"  Diferencia maxima vs barrido  : {max(abs(sol.bid - check.bid), abs(sol.ask - check.ask)):.4f}")
    print("\n  Descomposicion de la utilidad optima:")
    liq = params.pi_liquidity * ((sol.ask - params.s0) * (params.alpha - params.beta * (sol.ask - params.s0))
                                 + (params.s0 - sol.bid) * (params.alpha - params.beta * (params.s0 - sol.bid)))
    loss_ask = expected_informed_loss_ask(sol.ask, params)
    loss_bid = expected_informed_loss_bid(sol.bid, params)
    print(f"    + Ganancia esperada vs liquidez        : {liq:+.4f}")
    print(f"    - Perdida esperada vs informados (Ask) : {-params.pi_informed * loss_ask:+.4f}")
    print(f"    - Perdida esperada vs informados (Bid) : {-params.pi_informed * loss_bid:+.4f}")
    print(f"    = Utilidad esperada                    : {sol.expected_utility:+.4f}")

    # ---------------------------------------------------------------------- 3
    _rule("3. VALIDACION DEL SIMULADOR CONTRA LA TEORIA")
    print("  Sin forzar la ejecucion, el P&L medio por llegada debe converger a Pi(A,B).")
    val = validate_against_theory(sol.bid, sol.ask, params, n_trades=500_000)
    print(f"  P&L medio simulado ({val['n_trades']:,} llegadas) : {val['simulated_mean_pnl']:.4f}".replace(",", " "))
    print(f"  Utilidad esperada teorica                  : {val['theoretical_expected_utility']:.4f}")
    print(f"  Error absoluto                             : {val['abs_error']:.4f}")

    # ---------------------------------------------------------------------- 4
    regimes = {
        "Optimo":    (round(sol.bid, 2), round(sol.ask, 2)),
        "Estrecho":  (19.75, 20.05),
        "Amplio":    (18.40, 21.40),
    }

    _rule(f"4. SIMULACION DE {N_TRADES:,} TRADES POR REGIMEN (ejecucion forzada)".replace(",", " "))
    runs = {}
    print(f"  {'Regimen':<10} {'Bid':>7} {'Ask':>7} {'Spread':>7} "
          f"{'P&L total':>11} {'P&L/trade':>10} {'Inv. final':>11} {'Max |inv|':>10}")
    print("  " + "-" * 78)
    for name, (bid, ask) in regimes.items():
        rng = np.random.default_rng(DEFAULT_SEED)
        df = simulate_trades(bid, ask, n_trades=N_TRADES, params=params, rng=rng)
        runs[name] = df
        s = summarize_run(df)
        print(f"  {name:<10} {bid:>7.2f} {ask:>7.2f} {ask - bid:>7.2f} "
              f"{s['total_pnl']:>11.2f} {s['mean_pnl_per_arrival']:>10.4f} "
              f"{s['final_inventory']:>11d} {s['max_abs_inventory']:>10d}")

    print("\n  Origen del P&L (quien paga y quien cobra):")
    print(f"  {'Regimen':<10} {'P&L liquidez':>14} {'P&L informados':>16} "
          f"{'Trades informados':>19} {'Utilidad teorica':>18}")
    print("  " + "-" * 78)
    for name, (bid, ask) in regimes.items():
        s = summarize_run(runs[name])
        theo = expected_utility(ask, bid, params)
        print(f"  {name:<10} {s['pnl_from_liquidity']:>14.2f} {s['pnl_from_informed']:>16.2f} "
              f"{s['n_informed_trades']:>19d} {theo:>18.4f}")

    # ---------------------------------------------------------------------- 5
    _rule(f"5. MONTE CARLO: {MC_RUNS:,} CORRIDAS DE {MC_TRADES:,} TRADES".replace(",", " "))
    mc_results = {}
    mc_stats = {}
    print(f"  {'Regimen':<10} {'P&L promedio':>14} {'Desv. estandar':>16} "
          f"{'Prob. de perdida':>18} {'P5':>10} {'P95':>10}")
    print("  " + "-" * 78)
    for name, (bid, ask) in regimes.items():
        mc = run_monte_carlo(bid, ask, n_runs=MC_RUNS, n_trades=MC_TRADES,
                             params=params, seed=DEFAULT_SEED)
        mc_results[name] = mc["final_pnl"].to_numpy()
        st = monte_carlo_stats(mc)
        mc_stats[name] = st
        print(f"  {name:<10} {st['mean_pnl']:>14.2f} {st['std_pnl']:>16.2f} "
              f"{st['prob_loss']:>17.1%} {st['p05']:>10.2f} {st['p95']:>10.2f}")

    print(f"\n  Riesgo de inventario, promediado sobre las {MC_RUNS:,} corridas".replace(",", " "))
    print("  (los traders de liquidez son simetricos: toda la deriva la genera el flujo informado)")
    print(f"  {'Regimen':<10} {'Inv. medio':>12} {'E[deriva] teorica':>19} "
          f"{'|Inv| medio':>13} {'Max |inv| medio':>17} {'Trades inf.':>12}")
    print("  " + "-" * 78)
    for name, (bid, ask) in regimes.items():
        st = mc_stats[name]
        drift = expected_inventory_drift(bid, ask, params) * MC_TRADES
        print(f"  {name:<10} {st['mean_inventory']:>12.2f} {drift:>19.2f} "
              f"{st['mean_abs_inventory']:>13.2f} {st['mean_max_abs_inventory']:>17.2f} "
              f"{st['mean_informed_trades']:>12.1f}")

    # ---------------------------------------------------------------------- 6
    _rule("6. ANALISIS DE SENSIBILIDAD A pi_I")
    rows = sensitivity_to_pi_informed(PI_GRID, params)
    print(f"  {'pi_I':>6} {'Bid':>9} {'Ask':>9} {'Spread':>9} {'Utilidad':>11}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['pi_informed']:>6.2f} {r['bid']:>9.2f} {r['ask']:>9.2f} "
              f"{r['spread']:>9.2f} {r['expected_utility']:>11.2f}")
    zero = sensitivity_to_pi_informed([0.0], params)[0]
    print(f"\n  Referencia analitica pi_I = 0 (monopolista sin seleccion adversa):")
    print(f"    spread teorico  = alpha/beta = {params.zero_demand_spread:.2f}")
    print(f"    spread numerico = {zero['spread']:.2f}  (error = {abs(zero['spread'] - params.zero_demand_spread):.4f})")

    # ---------------------------------------------------------------------- 7
    _rule("7. FIGURAS")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("fig1_prob_ejecucion.png",
         lambda p: plots.plot_execution_probability(params, p)),
        ("fig2_pnl_acumulado.png",
         lambda p: plots.plot_cumulative_pnl(runs, p)),
        ("fig3_inventario.png",
         lambda p: plots.plot_inventory(runs, p)),
        ("fig4_histograma_montecarlo.png",
         lambda p: plots.plot_mc_histogram(mc_results, p)),
        ("fig5_sensibilidad_spread.png",
         lambda p: plots.plot_spread_sensitivity(rows, params, p)),
    ]
    for filename, builder in outputs:
        builder(FIGURES_DIR / filename)
        print(f"  guardada -> docs/figures/{filename}")

    print("\nEjecucion completa.\n")


if __name__ == "__main__":
    main()
