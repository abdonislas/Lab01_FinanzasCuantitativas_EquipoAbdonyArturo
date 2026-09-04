"""
Simulador de trades para el formador de mercado.

Idea general
------------
En cada "llegada" aparece un trader. Con probabilidad pi_informed es un
trader INFORMADO (conoce el valor verdadero P del activo); si no, es un
trader de LIQUIDEZ (no tiene informacion extra).

El precio "justo" contra el que se mide el P&L del formador de mercado (MM)
depende de con quien esta operando:

* Contra un informado, el precio justo es el valor verdadero P.
* Contra un trader de liquidez, el precio justo es S0 (el MM no sabe mas
  que eso).

Si el MM vende una unidad a X: gana X - (precio justo).
Si el MM compra una unidad a X: gana (precio justo) - X.

Con esta convencion el P&L promedio simulado deberia coincidir con la
formula teorica Pi(A, B) de model.py -- eso es justo lo que revisa
`validate_against_theory` mas abajo.

Ejecucion forzada vs. ejecucion realista
-----------------------------------------
El lab pide FORZAR un trade en cada llegada (force_execution=True, el
default): el trader de liquidez siempre opera, mitad compra / mitad vende,
sin importar que tan ancho este el spread. Es una simplificacion: en la
vida real un spread muy ancho casi no se ejecutaria. Por eso estos
resultados son rentabilidad POR TRADE, no por unidad de tiempo, y
favorecen de forma artificial a los spreads anchos (ver advertencia en el
README y en el notebook).

Con force_execution=False si se respeta pi_LB / pi_LS y el trader de
liquidez puede quedarse sin operar. Este modo solo se usa para la prueba
de consistencia en `validate_against_theory`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import ModelParams, execution_probability, expected_utility

# Semilla maestra del proyecto. Documentada en el README para reproducibilidad.
DEFAULT_SEED = 42

TRADER_INFORMED = "informado"
TRADER_LIQUIDITY = "liquidez"
SIDE_MM_SELLS = "venta_mm"   # el trader compra al Ask -> el MM vende
SIDE_MM_BUYS = "compra_mm"   # el trader vende al Bid  -> el MM compra
SIDE_NONE = "sin_trade"


def simulate_trades(bid: float,
                    ask: float,
                    n_trades: int = 10_000,
                    params: ModelParams | None = None,
                    rng: np.random.Generator | None = None,
                    force_execution: bool = True) -> pd.DataFrame:
    """Simula n_trades llegadas contra un par de cotizaciones (bid, ask) fijo.

    Regresa un DataFrame con una fila por llegada: trade_id, trader_type,
    side, price, true_value, pnl, inventory_change, executed, cum_pnl,
    inventory.
    """
    params = params or ModelParams()
    rng = rng or np.random.default_rng(DEFAULT_SEED)

    # 1) Tirar todos los dados que hacen falta, de una vez.
    is_informed = rng.random(n_trades) < params.pi_informed
    true_value = params.value_dist.rvs(size=n_trades, random_state=rng)
    liquidity_wants_to_buy = rng.random(n_trades) < 0.5

    # 2) Preparar las columnas de salida, todo en "no operado" por defecto.
    n = n_trades
    trader_type = np.where(is_informed, TRADER_INFORMED, TRADER_LIQUIDITY)
    side = np.full(n, SIDE_NONE, dtype=object)
    price = np.zeros(n)
    pnl = np.zeros(n)
    inventory_change = np.zeros(n, dtype=int)

    # 3) Decidir, trade por trade, quien opera y de que lado.
    #    Un informado solo ataca la cotizacion si le conviene: compra al
    #    Ask si el valor real es mayor al Ask, vende al Bid si es menor.
    informed_buys = is_informed & (true_value > ask)
    informed_sells = is_informed & (true_value < bid)

    if force_execution:
        # El trader de liquidez siempre opera (asi lo pide el enunciado),
        # 50/50 compra o vende, sin mirar el spread.
        liquidity_buys = (~is_informed) & liquidity_wants_to_buy
        liquidity_sells = (~is_informed) & (~liquidity_wants_to_buy)
    else:
        # Version realista: cada lado se ejecuta con su propia
        # probabilidad pi_LB(s) / pi_LS(s), y puede no ejecutarse ninguno.
        p_exec_ask = execution_probability(ask - params.s0, params)
        p_exec_bid = execution_probability(params.s0 - bid, params)
        exec_draw = rng.random(n_trades)
        liquidity_buys = (~is_informed) & (exec_draw < p_exec_ask)
        liquidity_sells = (~is_informed) & (exec_draw >= p_exec_ask) & \
            (exec_draw < p_exec_ask + p_exec_bid)

    # 4) Llenar precio, P&L y cambio de inventario para cada grupo.
    #    Informados: el precio justo es el valor verdadero (true_value).
    side[informed_buys] = SIDE_MM_SELLS
    price[informed_buys] = ask
    pnl[informed_buys] = ask - true_value[informed_buys]
    inventory_change[informed_buys] = -1

    side[informed_sells] = SIDE_MM_BUYS
    price[informed_sells] = bid
    pnl[informed_sells] = true_value[informed_sells] - bid
    inventory_change[informed_sells] = 1

    #    Liquidez: el precio justo es S0 (el MM no tiene informacion extra).
    side[liquidity_buys] = SIDE_MM_SELLS
    price[liquidity_buys] = ask
    pnl[liquidity_buys] = ask - params.s0
    inventory_change[liquidity_buys] = -1

    side[liquidity_sells] = SIDE_MM_BUYS
    price[liquidity_sells] = bid
    pnl[liquidity_sells] = params.s0 - bid
    inventory_change[liquidity_sells] = 1

    # 5) Armar el DataFrame y las columnas acumuladas (Figuras 2 y 3).
    df = pd.DataFrame({
        "trade_id": np.arange(1, n_trades + 1),
        "trader_type": trader_type,
        "side": side,
        "price": price,
        "true_value": true_value,
        "pnl": pnl,
        "inventory_change": inventory_change,
        "executed": side != SIDE_NONE,
    })
    df["cum_pnl"] = df["pnl"].cumsum()
    df["inventory"] = df["inventory_change"].cumsum()
    return df


def summarize_run(df: pd.DataFrame) -> dict:
    """Junta los numeros importantes de una corrida de simulate_trades."""
    executed = df[df["executed"]]
    informed = executed[executed["trader_type"] == TRADER_INFORMED]
    liquidity = executed[executed["trader_type"] == TRADER_LIQUIDITY]

    return {
        "n_arrivals": len(df),
        "n_trades": int(len(executed)),
        "fill_rate": len(executed) / len(df),
        "total_pnl": float(df["pnl"].sum()),
        "mean_pnl_per_arrival": float(df["pnl"].mean()),
        "std_pnl_per_arrival": float(df["pnl"].std(ddof=1)),
        "pnl_from_liquidity": float(liquidity["pnl"].sum()),
        "pnl_from_informed": float(informed["pnl"].sum()),
        "n_informed_trades": int(len(informed)),
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
    """Corre n_runs simulaciones independientes y guarda el resultado final de cada una.

    Usamos SeedSequence para repartir n_runs semillas hijas a partir de una
    sola semilla maestra: asi las 1000 corridas quedan de verdad
    independientes entre si (no encadenadas) y el conjunto sigue siendo
    reproducible.

    Regresa un DataFrame con una fila por corrida: final_pnl,
    final_inventory, max_abs_inventory, n_informed_trades.
    """
    params = params or ModelParams()
    child_seeds = np.random.SeedSequence(seed).spawn(n_runs)

    records = []
    for child_seed in child_seeds:
        rng = np.random.default_rng(child_seed)
        df = simulate_trades(bid, ask, n_trades=n_trades, params=params,
                             rng=rng, force_execution=force_execution)
        is_informed_trade = (df["trader_type"] == TRADER_INFORMED) & df["executed"]
        records.append({
            "final_pnl": float(df["pnl"].sum()),
            "final_inventory": int(df["inventory_change"].sum()),
            "max_abs_inventory": int(df["inventory"].abs().max()),
            "n_informed_trades": int(is_informed_trade.sum()),
        })
    return pd.DataFrame(records)


def monte_carlo_stats(mc: pd.DataFrame) -> dict:
    """Las tres cifras que pide el enunciado (media, desviacion, prob. de
    perdida) mas algo de inventario, para comparar regimenes."""
    final_pnl = mc["final_pnl"].to_numpy()
    final_inventory = mc["final_inventory"].to_numpy()

    return {
        "mean_pnl": float(np.mean(final_pnl)),
        "std_pnl": float(np.std(final_pnl, ddof=1)),
        "prob_loss": float(np.mean(final_pnl < 0.0)),
        "p05": float(np.percentile(final_pnl, 5)),
        "p95": float(np.percentile(final_pnl, 95)),
        "mean_inventory": float(np.mean(final_inventory)),
        "mean_abs_inventory": float(np.mean(np.abs(final_inventory))),
        "std_inventory": float(np.std(final_inventory, ddof=1)),
        "mean_max_abs_inventory": float(mc["max_abs_inventory"].mean()),
        "mean_informed_trades": float(mc["n_informed_trades"].mean()),
    }


def expected_inventory_drift(bid: float, ask: float,
                             params: ModelParams | None = None) -> float:
    """Hacia donde se va el inventario en promedio, sin tener que simular.

    Los de liquidez son 50/50 y se cancelan en promedio. Toda la deriva
    sistematica viene de los informados: compran cuando P > Ask (inventario
    del MM baja 1) y venden cuando P < Bid (inventario sube 1):

        E[cambio de inventario] = pi_informed * (Prob(P < Bid) - Prob(P > Ask))
    """
    params = params or ModelParams()
    dist = params.value_dist
    prob_informed_sells = dist.cdf(bid)   # Prob(P < Bid)
    prob_informed_buys = dist.sf(ask)     # Prob(P > Ask)
    return float(params.pi_informed * (prob_informed_sells - prob_informed_buys))


def validate_against_theory(bid: float,
                            ask: float,
                            params: ModelParams | None = None,
                            n_trades: int = 500_000,
                            seed: int = DEFAULT_SEED) -> dict:
    """Prueba de consistencia entre este archivo y model.py.

    Corre con force_execution=False (respetando la probabilidad real de
    ejecucion) y con muchos trades, para que el P&L promedio simulado
    converja a la utilidad esperada teorica Pi(A, B). Si abs_error sale
    chico, model.py y simulation.py estan describiendo el mismo mecanismo.
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
