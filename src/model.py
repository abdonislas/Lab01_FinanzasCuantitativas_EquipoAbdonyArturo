"""
Modelo de Copeland y Galai (1983): cotizaciones optimas de un formador de mercado.

Contiene la funcion de utilidad esperada por trader que llega, sus componentes
(ganancia frente a liquidez, perdida esperada frente a informados) y la rutina
de optimizacion de las cotizaciones Bid y Ask.

Formulacion implementada
------------------------
    Pi(A, B) = pi_L * [ pi_LB(A - S0) * (A - S0) + pi_LS(S0 - B) * (S0 - B) ]
             - pi_I * [ int_A^inf (P - A) f(P) dP + int_0^B (B - P) f(P) dP ]

con f(P) la densidad Erlang(K=60, lambda=3) del valor verdadero del activo.

Supuestos declarados
--------------------
1. El formador de mercado cotiza una unidad por lado y cada trader que llega
   demanda a lo mas una unidad.
2. pi_L = 1 - pi_I: solo existen dos tipos de trader (liquidez e informado).
3. La demanda no informada es identica en ambos lados:
   pi_LB(s) = pi_LS(s) = max(0, alpha - beta * s), con alpha = 0.50, beta = 0.08.
   El truncamiento en cero es obligatorio: sin el, la "probabilidad" seria
   negativa para s > 6.25 y la funcion objetivo dejaria de tener sentido.
4. El trader informado observa P sin ruido y solo opera cuando le conviene:
   compra al Ask si P > A y vende al Bid si P < B. De ahi los limites de las
   integrales de perdida.
5. El formador de mercado es neutral al riesgo y no gestiona inventario: el
   modelo no penaliza posiciones acumuladas (ver limitaciones en el README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats

# Tolerancia usada para truncar las colas de la Erlang al integrar.
# La masa de probabilidad despreciada por lado es menor a TAIL_TOL, es decir
# el error de truncamiento del valor esperado es despreciable frente a la
# precision de dos decimales que exige el reporte.
TAIL_TOL = 1e-13


@dataclass(frozen=True)
class ModelParams:
    """Parametros del caso base del laboratorio (Seccion 3.1 del enunciado)."""

    s0: float = 19.90          # Precio de referencia
    k: int = 60                # Parametro de forma de la Erlang
    lam: float = 3.0           # Parametro de tasa de la Erlang
    pi_informed: float = 0.40  # Probabilidad de que el trader este informado
    alpha: float = 0.50        # Intercepto de la demanda no informada
    beta: float = 0.08         # Sensibilidad de la demanda al spread por lado

    @property
    def pi_liquidity(self) -> float:
        """Probabilidad de que el trader sea de liquidez: pi_L = 1 - pi_I."""
        return 1.0 - self.pi_informed

    @property
    def value_dist(self) -> stats.rv_continuous:
        """Distribucion del valor verdadero P ~ Erlang(K, lambda).

        `scipy.stats.erlang` se parametriza con forma `a = K` y `scale = 1/lambda`,
        de modo que E[P] = K / lambda = 20.0 y sd(P) = sqrt(K) / lambda = 2.582.
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
    """Probabilidad de ejecucion de un trader de liquidez.

    pi_LB(s) = pi_LS(s) = max(0, alpha - beta * s)

    Parameters
    ----------
    spread_side : distancia entre la cotizacion y S0 (semi-spread), s >= 0.
    params : parametros del modelo.

    Returns
    -------
    Probabilidad en [0, alpha]. Nunca negativa.
    """
    s = np.asarray(spread_side, dtype=float)
    prob = np.maximum(params.alpha - params.beta * s, 0.0)
    return float(prob) if prob.ndim == 0 else prob


def expected_informed_loss_ask(ask: float, params: ModelParams) -> float:
    """Perdida esperada frente a informados del lado de la venta.

    Calcula int_A^inf (P - A) f(P) dP mediante `scipy.integrate.quad`.
    El limite superior infinito se sustituye por el cuantil 1 - TAIL_TOL de la
    Erlang (45.2198). La contribucion descartada vale 2.24e-12 en el optimo,
    diez ordenes de magnitud por debajo de los dos decimales que exige el
    reporte. Es una medida defensiva con error acotado y no la correccion de
    una falla observada: en el rango de cotizaciones relevante `quad` con
    limite np.inf devuelve el mismo valor (diferencia relativa < 1.7e-11).
    """
    dist = params.value_dist
    upper = float(dist.ppf(1.0 - TAIL_TOL))
    if ask >= upper:
        return 0.0
    value, _ = integrate.quad(lambda p: (p - ask) * dist.pdf(p),
                              ask, upper, limit=200)
    return float(value)


def expected_informed_loss_bid(bid: float, params: ModelParams) -> float:
    """Perdida esperada frente a informados del lado de la compra.

    Calcula int_0^B (B - P) f(P) dP mediante `scipy.integrate.quad`.
    El limite inferior 0 se sustituye por el cuantil TAIL_TOL de la Erlang
    (6.4351), donde la densidad es numericamente nula. La contribucion
    descartada vale 1.02e-12 en el optimo. Igual que del lado Ask, es una
    medida defensiva con error acotado: integrando desde 0, `quad` devuelve
    el mismo valor (diferencia relativa < 1.4e-11).
    """
    dist = params.value_dist
    lower = float(dist.ppf(TAIL_TOL))
    if bid <= lower:
        return 0.0
    value, _ = integrate.quad(lambda p: (bid - p) * dist.pdf(p),
                              lower, bid, limit=200)
    return float(value)


def expected_informed_loss(ask: float, bid: float, params: ModelParams) -> float:
    """Perdida esperada total frente a traders informados (sin ponderar por pi_I)."""
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
    """Maximiza Pi(A, B) minimizando -Pi(A, B) con `scipy.optimize.minimize`.

    Restricciones: B en (0, S0] y A en [S0, inf). El limite superior infinito
    del Ask se sustituye por `ask_upper` (por defecto S0 mas dos veces el
    semi-spread que anula la demanda). Es un limite no vinculante: mas alla de
    S0 + alpha/beta la ganancia de liquidez es exactamente cero y la utilidad
    solo puede decrecer, por lo que ningun optimo puede vivir en esa region.
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
    """Optimo por barrido fino, usado solo para verificar `optimize_quotes`.

    La utilidad es separable en A y B, asi que basta un barrido unidimensional
    por lado. No sustituye a `scipy.optimize.minimize`: sirve como prueba
    independiente de que el optimizador no quedo atrapado en una region plana.
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
    """Reoptimiza las cotizaciones para cada valor de pi_I.

    Returns
    -------
    list[dict] con pi_informed, bid, ask, spread y expected_utility.
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
