"""
Modelo de Copeland y Galai (1983): cotizaciones optimas de un formador de mercado.

Aqui vive la funcion de utilidad esperada por trader que llega, sus dos
componentes (ganancia frente a liquidez, perdida frente a informados) y la
optimizacion de las cotizaciones Bid y Ask.

Formulacion implementada
------------------------
    Pi(A, B) = pi_L * [ pi_LB(A - S0) * (A - S0) + pi_LS(S0 - B) * (S0 - B) ]
             - pi_I * [ int_A^inf (P - A) f(P) dP + int_0^B (B - P) f(P) dP ]

con f(P) la densidad Erlang(K=60, lambda=3) del valor verdadero del activo.

Supuestos del modelo
---------------------
1. El MM cotiza una unidad por lado; cada trader que llega pide a lo mas una.
2. Solo hay dos tipos de trader: informado y de liquidez (pi_L = 1 - pi_I).
3. La demanda no informada es igual en ambos lados:
   pi_LB(s) = pi_LS(s) = max(0, alpha - beta * s), alpha = 0.50, beta = 0.08.
   El truncamiento en cero es necesario: sin el, para s > 6.25 la
   "probabilidad" saldria negativa y la funcion objetivo dejaria de tener
   sentido.
4. El informado ve P sin ruido y solo opera cuando le conviene: compra al
   Ask si P > A, vende al Bid si P < B. De ahi salen los limites de las
   integrales de perdida.
5. El MM es neutral al riesgo y no gestiona inventario -- el modelo no
   penaliza posiciones acumuladas (limitacion que se discute en el README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats

# Tolerancia para truncar las colas de la Erlang al integrar. La masa de
# probabilidad que se pierde por lado es menor a esto, muy por debajo de
# los dos decimales que exige el reporte.
TAIL_TOL = 1e-13


@dataclass(frozen=True)
class ModelParams:
    """Parametros del caso base (Seccion 3.1 del enunciado)."""

    s0: float = 19.90          # Precio de referencia
    k: int = 60                # Forma de la Erlang
    lam: float = 3.0           # Tasa de la Erlang
    pi_informed: float = 0.40  # Probabilidad de trader informado
    alpha: float = 0.50        # Intercepto de la demanda no informada
    beta: float = 0.08         # Sensibilidad de la demanda al spread por lado

    @property
    def pi_liquidity(self) -> float:
        """pi_L = 1 - pi_I."""
        return 1.0 - self.pi_informed

    @property
    def value_dist(self) -> stats.rv_continuous:
        """Distribucion del valor verdadero P ~ Erlang(K, lambda).

        scipy.stats.erlang se parametriza con forma a=K y scale=1/lambda,
        asi que E[P] = K/lambda = 20.0 y sd(P) = sqrt(K)/lambda = 2.582.
        """
        return stats.erlang(self.k, scale=1.0 / self.lam)

    @property
    def zero_demand_spread(self) -> float:
        """Semi-spread a partir del cual la demanda no informada se anula."""
        return self.alpha / self.beta


@dataclass(frozen=True)
class QuoteSolution:
    """Resultado de la optimizacion de cotizaciones."""

    bid: float
    ask: float
    expected_utility: float
    converged: bool
    message: str

    @property
    def spread(self) -> float:
        """Spread total cotizado (Ask - Bid)."""
        return self.ask - self.bid


def execution_probability(spread_side: np.ndarray | float,
                          params: ModelParams) -> np.ndarray | float:
    """Probabilidad de ejecucion de un trader de liquidez: pi_LB(s) = pi_LS(s) = max(0, alpha - beta*s)."""
    s = np.asarray(spread_side, dtype=float)
    prob = np.maximum(params.alpha - params.beta * s, 0.0)
    return float(prob) if prob.ndim == 0 else prob


def expected_informed_loss_ask(ask: float, params: ModelParams) -> float:
    """Perdida esperada frente a informados, lado venta: integral de (P-A)*f(P) de A a infinito.

    Se integra con quad hasta un cuantil muy cercano a 1 en vez de hasta
    infinito literal, porque quad no acepta infinito directamente y la cola
    que se recorta pesa menos que TAIL_TOL.
    """
    dist = params.value_dist
    upper = float(dist.ppf(1.0 - TAIL_TOL))
    if ask >= upper:
        return 0.0
    value, _ = integrate.quad(lambda p: (p - ask) * dist.pdf(p),
                              ask, upper, limit=200)
    return float(value)


def expected_informed_loss_bid(bid: float, params: ModelParams) -> float:
    """Perdida esperada frente a informados, lado compra: integral de (B-P)*f(P) de 0 a B.

    Igual que del lado Ask: se integra desde un cuantil muy cercano a 0 en
    vez de desde 0 literal, porque ahi la densidad ya es numericamente nula.
    """
    dist = params.value_dist
    lower = float(dist.ppf(TAIL_TOL))
    if bid <= lower:
        return 0.0
    value, _ = integrate.quad(lambda p: (bid - p) * dist.pdf(p),
                              lower, bid, limit=200)
    return float(value)


def expected_informed_loss(ask: float, bid: float, params: ModelParams) -> float:
    """Perdida esperada total frente a informados (sin ponderar por pi_I)."""
    return (expected_informed_loss_ask(ask, params)
            + expected_informed_loss_bid(bid, params))


def expected_liquidity_gain(ask: float, bid: float, params: ModelParams) -> float:
    """Ganancia esperada frente a traders de liquidez (sin ponderar por pi_L)."""
    s_ask = ask - params.s0
    s_bid = params.s0 - bid
    return (execution_probability(s_ask, params) * s_ask
            + execution_probability(s_bid, params) * s_bid)


def expected_utility(ask: float, bid: float, params: ModelParams) -> float:
    """Utilidad esperada Pi(A, B) por trader que llega al mercado."""
    return (params.pi_liquidity * expected_liquidity_gain(ask, bid, params)
            - params.pi_informed * expected_informed_loss(ask, bid, params))


def negative_expected_utility(x: np.ndarray, params: ModelParams) -> float:
    """Funcion objetivo a minimizar: -Pi(A, B), con x = [A, B]."""
    ask, bid = float(x[0]), float(x[1])
    return -expected_utility(ask, bid, params)


def optimize_quotes(params: ModelParams | None = None,
                    ask_upper: float | None = None) -> QuoteSolution:
    """Encuentra el Bid y Ask que maximizan Pi(A, B), minimizando -Pi(A, B).

    Restricciones: B en (0, S0] y A en [S0, inf). El "infinito" del Ask se
    reemplaza por `ask_upper` (S0 mas dos veces el semi-spread que anula la
    demanda). No cambia el resultado: mas alla de ese punto la ganancia de
    liquidez ya es cero, asi que el optimo nunca puede caer ahi.
    """
    params = params or ModelParams()
    if ask_upper is None:
        ask_upper = params.s0 + 2.0 * params.zero_demand_spread

    bounds = [(params.s0, ask_upper), (1e-6, params.s0)]
    x0 = np.array([params.s0 + 1.0, params.s0 - 1.0])

    result = optimize.minimize(
        negative_expected_utility,
        x0,
        args=(params,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
    )

    ask, bid = float(result.x[0]), float(result.x[1])
    return QuoteSolution(
        bid=bid,
        ask=ask,
        expected_utility=float(-result.fun),
        converged=bool(result.success),
        message=str(result.message),
    )


def optimize_quotes_grid(params: ModelParams | None = None,
                         n_points: int = 4001) -> QuoteSolution:
    """Optimo por barrido fino, usado solo para verificar optimize_quotes.

    La utilidad se puede separar en una parte que solo depende de A y otra
    que solo depende de B, asi que basta barrer cada lado por separado. No
    reemplaza a optimize_quotes: es una segunda forma de llegar al mismo
    numero, para confirmar que el optimizador no se quedo atorado.
    """
    params = params or ModelParams()
    grid = np.linspace(0.0, params.zero_demand_spread, n_points)

    ask_values = [params.pi_liquidity * execution_probability(s, params) * s
                  - params.pi_informed * expected_informed_loss_ask(params.s0 + s, params)
                  for s in grid]
    bid_values = [params.pi_liquidity * execution_probability(s, params) * s
                  - params.pi_informed * expected_informed_loss_bid(params.s0 - s, params)
                  for s in grid]

    ask = params.s0 + grid[int(np.argmax(ask_values))]
    bid = params.s0 - grid[int(np.argmax(bid_values))]
    return QuoteSolution(
        bid=bid,
        ask=ask,
        expected_utility=expected_utility(ask, bid, params),
        converged=True,
        message="grid search",
    )


def sensitivity_to_pi_informed(pi_values, params: ModelParams | None = None):
    """Reoptimiza las cotizaciones para cada valor de pi_I en pi_values.

    Regresa una lista de diccionarios con pi_informed, bid, ask, spread y
    expected_utility -- una fila por cada valor probado.
    """
    base = params or ModelParams()
    rows = []
    for pi in pi_values:
        p = ModelParams(s0=base.s0, k=base.k, lam=base.lam,
                        pi_informed=float(pi), alpha=base.alpha, beta=base.beta)
        sol = optimize_quotes(p)
        rows.append({
            "pi_informed": float(pi),
            "bid": sol.bid,
            "ask": sol.ask,
            "spread": sol.spread,
            "expected_utility": sol.expected_utility,
        })
    return rows
