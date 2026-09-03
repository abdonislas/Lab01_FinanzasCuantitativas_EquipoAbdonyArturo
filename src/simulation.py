"""
Simulador de trades para el formador de mercado.

Convencion de P&L
-----------------
Cada operacion se valora contra el valor fundamental relevante para la
contraparte:

* Si el formador VENDE una unidad al precio X:  P&L = X - V,  inventario -1
* Si el formador COMPRA una unidad al precio X: P&L = V - X,  inventario +1

donde V = S0 frente a un trader de liquidez (el formador no tiene informacion
adicional) y V = P frente a un trader informado (el informado conoce el valor
verdadero). Con esta convencion el valor esperado del P&L simulado reproduce
exactamente la funcion de utilidad Pi(A, B) de `model.py`, lo que se usa como
prueba de consistencia en `validate_against_theory`.

Ejecucion forzada
-----------------
El enunciado exige forzar un trade en cada iteracion. Por eso, con
`force_execution=True` (valor por defecto) el trader de liquidez SIEMPRE opera
e ignoramos pi_LB(s) y pi_LS(s). Consecuencia que hay que declarar de forma
explicita: el resultado mide rentabilidad POR TRADE, no por unidad de tiempo,
y por lo tanto favorece artificialmente a los spreads amplios, que en un
mercado real casi nunca se ejecutarian.

Con `force_execution=False` la probabilidad de ejecucion si se respeta y el
promedio simulado converge a Pi(A, B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import ModelParams, execution_probability, expected_utility

# Semilla maestra del proyecto. Documentada en el README.
DEFAULT_SEED = 42

TRADER_INFORMED = "informado"
TRADER_LIQUIDITY = "liquidez"
SIDE_MM_SELLS = "venta_mm"   # el trader compra al Ask
SIDE_MM_BUYS = "compra_mm"   # el trader vende al Bid
SIDE_NONE = "sin_trade"


def simulate_trades(bid: float,
                    ask: float,
                    n_trades: int = 10_000,
                    params: ModelParams | None = None,
                    rng: np.random.Generator | None = None,
                    force_execution: bool = True) -> pd.DataFrame:
    """Simula `n_trades` llegadas de traders contra un par de cotizaciones fijo.

    Returns
    -------
    DataFrame con una fila por llegada y columnas:
        trade_id, trader_type, side, price, true_value, pnl,
        inventory_change, executed, cum_pnl, inventory.
    """
    params = params or ModelParams()
    rng = rng or np.random.default_rng(DEFAULT_SEED)

    s_ask = ask - params.s0
    s_bid = params.s0 - bid
    p_exec_ask = execution_probability(s_ask, params)
    p_exec_bid = execution_probability(s_bid, params)

    is_informed = rng.random(n_trades) < params.pi_informed
    true_value = params.value_dist.rvs(size=n_trades, random_state=rng)
    buys = rng.random(n_trades) < 0.5   # direccion forzada del trader de liquidez
    exec_draw = rng.random(n_trades)    # decide compra/venta/no-trade si no se fuerza

    trader_type = np.where(is_informed, TRADER_INFORMED, TRADER_LIQUIDITY)
    side = np.full(n_trades, SIDE_NONE, dtype=object)
    price = np.zeros(n_trades)
    pnl = np.zeros(n_trades)
    inventory_change = np.zeros(n_trades, dtype=int)

    # --- Traders informados: operan solo si la cotizacion es atacable ---------
    inf_buy = is_informed & (true_value > ask)    # el informado compra al Ask
    inf_sell = is_informed & (true_value < bid)   # el informado vende al Bid

    side[inf_buy] = SIDE_MM_SELLS
    price[inf_buy] = ask
    pnl[inf_buy] = ask - true_value[inf_buy]      # siempre negativo
    inventory_change[inf_buy] = -1

    side[inf_sell] = SIDE_MM_BUYS
    price[inf_sell] = bid
    pnl[inf_sell] = true_value[inf_sell] - bid    # siempre negativo
    inventory_change[inf_sell] = 1

    # --- Traders de liquidez -------------------------------------------------
    # pi_LB(s) y pi_LS(s) son, respectivamente, la probabilidad de que un trader
    # de liquidez COMPRE al Ask y de que VENDA al Bid. En s = 0 suman 1 (siempre
    # opera); al ampliarse el spread ambas caen y aparece la posibilidad de que
    # el trader se retire sin operar.
    liq = ~is_informed
    if force_execution:
        # Ejecucion forzada: se ignoran pi_LB y pi_LS. Como pi_LB = pi_LS, la
        # direccion condicionada a operar es 50/50.
        liq_buy = liq & buys         # el trader compra al Ask -> el MM vende
        liq_sell = liq & ~buys       # el trader vende al Bid  -> el MM compra
    else:
        liq_buy = liq & (exec_draw < p_exec_ask)
        liq_sell = liq & (exec_draw >= p_exec_ask) & (exec_draw < p_exec_ask + p_exec_bid)

    side[liq_buy] = SIDE_MM_SELLS
    price[liq_buy] = ask
    pnl[liq_buy] = ask - params.s0
    inventory_change[liq_buy] = -1

    side[liq_sell] = SIDE_MM_BUYS
    price[liq_sell] = bid
    pnl[liq_sell] = params.s0 - bid
    inventory_change[liq_sell] = 1

    executed = side != SIDE_NONE

    df = pd.DataFrame({
        "trade_id": np.arange(1, n_trades + 1),
        "trader_type": trader_type,
        "side": side,
        "price": price,
        "true_value": true_value,
        "pnl": pnl,
        "inventory_change": inventory_change,
        "executed": executed,
    })
    df["cum_pnl"] = df["pnl"].cumsum()
    df["inventory"] = df["inventory_change"].cumsum()
    return df


def summarize_run(df: pd.DataFrame) -> dict:
    """Estadisticos descriptivos de una corrida de `simulate_trades`."""
    executed = df[df["executed"]]
    informed = executed[executed["trader_type"] == TRADER_INFORMED]
    liquidity = executed[executed["trader_type"] == TRADER_LIQUIDITY]
    return {
        "n_arrivals": len(df),
        "n_trades": int(executed.shape[0]),
        "fill_rate": executed.shape[0] / len(df),
        "total_pnl": float(df["pnl"].sum()),
        "mean_pnl_per_arrival": float(df["pnl"].mean()),
        "std_pnl_per_arrival": float(df["pnl"].std(ddof=1)),
        "pnl_from_liquidity": float(liquidity["pnl"].sum()),
        "pnl_from_informed": float(informed["pnl"].sum()),
        "n_informed_trades": int(informed.shape[0]),
        "final_inventory": int(df["inventory_change"].sum()),
        "max_abs_inventory": int(df["inventory"].abs().max()),
    }


def run_monte_carlo(bid: float,
                    ask: float,
                    n_runs: int = 1_000,
                    n_trades: int = 1_000,
                    params: ModelParams | None = None,
                    seed: int = DEFAULT_SEED,
                    force_execution: bool = True) -> pd.DataFrame:
    """Ejecuta `n_runs` corridas independientes de `n_trades` trades cada una.

    Se usa una `SeedSequence` derivada de `seed` para que cada corrida tenga un
    flujo aleatorio independiente y el conjunto siga siendo reproducible.

    Returns
    -------
    DataFrame con una fila por corrida: final_pnl, final_inventory,
    max_abs_inventory y n_informed_trades.
    """
    params = params or ModelParams()
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    records = []
    for child in seeds:
        rng = np.random.default_rng(child)
        df = simulate_trades(bid, ask, n_trades=n_trades, params=params,
                             rng=rng, force_execution=force_execution)
        records.append({
            "final_pnl": float(df["pnl"].sum()),
            "final_inventory": int(df["inventory_change"].sum()),
            "max_abs_inventory": int(df["inventory"].abs().max()),
            "n_informed_trades": int(((df["trader_type"] == TRADER_INFORMED)
                                      & df["executed"]).sum()),
        })
    return pd.DataFrame(records)


def monte_carlo_stats(mc: pd.DataFrame) -> dict:
    """Estadisticos del Monte Carlo.

    Incluye las tres cifras exigidas por el enunciado (P&L promedio, desviacion
    estandar y probabilidad de perdida) mas metricas de inventario, necesarias
    para comparar el desbalance entre regimenes sin depender de una sola
    trayectoria.
    """
    finals = mc["final_pnl"].to_numpy()
    inv = mc["final_inventory"].to_numpy()
    return {
        "mean_pnl": float(np.mean(finals)),
        "std_pnl": float(np.std(finals, ddof=1)),
        "prob_loss": float(np.mean(finals < 0.0)),
        "p05": float(np.percentile(finals, 5)),
        "p95": float(np.percentile(finals, 95)),
        "mean_inventory": float(np.mean(inv)),
        "mean_abs_inventory": float(np.mean(np.abs(inv))),
        "std_inventory": float(np.std(inv, ddof=1)),
        "mean_max_abs_inventory": float(mc["max_abs_inventory"].mean()),
        "mean_informed_trades": float(mc["n_informed_trades"].mean()),
    }


def expected_inventory_drift(bid: float, ask: float,
                             params: ModelParams | None = None) -> float:
    """Deriva esperada del inventario por llegada, bajo ejecucion forzada.

    Los traders de liquidez son simetricos y no aportan deriva: su contribucion
    esperada es cero. Toda la deriva proviene del flujo informado, que compra
    cuando P > A (inventario -1) y vende cuando P < B (inventario +1):

        E[dI] = pi_I * ( Pr(P < B) - Pr(P > A) )

    Separar la deriva del ruido es lo que permite responder que regimen acumula
    mas desbalance sistematico, en lugar de comparar un unico camino aleatorio.
    """
    params = params or ModelParams()
    dist = params.value_dist
    return float(params.pi_informed * (dist.cdf(bid) - dist.sf(ask)))


def validate_against_theory(bid: float,
                            ask: float,
                            params: ModelParams | None = None,
                            n_trades: int = 500_000,
                            seed: int = DEFAULT_SEED) -> dict:
    """Comprueba que el simulador reproduce Pi(A, B) cuando NO se fuerza la ejecucion.

    Es la prueba de consistencia entre `model.py` y `simulation.py`: si ambas
    piezas describen el mismo mecanismo, el P&L medio por llegada debe converger
    a la utilidad esperada teorica.
    """
    params = params or ModelParams()
    rng = np.random.default_rng(seed)
    df = simulate_trades(bid, ask, n_trades=n_trades, params=params,
                         rng=rng, force_execution=False)
    simulated = float(df["pnl"].mean())
    theoretical = expected_utility(ask, bid, params)
    return {
        "simulated_mean_pnl": simulated,
        "theoretical_expected_utility": theoretical,
        "abs_error": abs(simulated - theoretical),
        "n_trades": n_trades,
    }
