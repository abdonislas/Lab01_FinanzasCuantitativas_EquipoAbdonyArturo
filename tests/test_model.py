"""
Pruebas del modelo de Copeland-Galai.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Permite importar src/ sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (ModelParams, execution_probability,
                       expected_informed_loss_ask,
                       expected_informed_loss_bid, optimize_quotes)


@pytest.fixture
def parametros() -> ModelParams:
    return ModelParams()


# --------------------------------------------------------------------- 1
def test_probabilidad_ejecucion_nunca_es_negativa(parametros):

    spreads = np.linspace(0.0, 3.0 * parametros.zero_demand_spread, 500)
    probabilidades = execution_probability(spreads, parametros)

    assert np.all(probabilidades >= 0.0)
    # Un valor escalar mas alla del punto de demanda nula tambien da cero.
    assert execution_probability(parametros.zero_demand_spread + 5.0, parametros) == 0.0
    # Y en s = 0 la probabilidad es exactamente alpha.
    assert execution_probability(0.0, parametros) == pytest.approx(parametros.alpha)


# --------------------------------------------------------------------- 2
def test_perdida_esperada_es_decreciente_en_ask(parametros):
    """La perdida esperada frente a informados por el lado Ask debe ser
    decreciente en A: entre mas alto cotiza el market maker, menos
    veces le compra un informado y menos pierde cuando lo hace.
    """
    asks = [parametros.s0, parametros.s0 + 1.0, parametros.s0 + 2.0, parametros.s0 + 4.0]
    perdidas = [expected_informed_loss_ask(a, parametros) for a in asks]

    for menor, mayor in zip(perdidas, perdidas[1:]):
        assert mayor < menor

    # la perdida por el lado Bid es creciente en B.
    bids = [parametros.s0 - 4.0, parametros.s0 - 2.0, parametros.s0 - 1.0, parametros.s0]
    perdidas_bid = [expected_informed_loss_bid(b, parametros) for b in bids]
    for menor, mayor in zip(perdidas_bid, perdidas_bid[1:]):
        assert mayor > menor


# --------------------------------------------------------------------- 3
def test_spread_de_monopolista_sin_traders_informados(parametros):
    """Sin traders informados, el spread optimo se puede calcular:
    3.125 por lado (alpha/2beta), 6.25 en total (alpha/beta)."""
    sin_informados = ModelParams(pi_informed=0.0)
    solucion = optimize_quotes(sin_informados)

    semi_spread_teorico = parametros.alpha / (2.0 * parametros.beta)
    spread_total_teorico = 0.50 / 0.08

    assert solucion.ask - sin_informados.s0 == pytest.approx(semi_spread_teorico, abs=1e-3)
    assert sin_informados.s0 - solucion.bid == pytest.approx(semi_spread_teorico, abs=1e-3)
    assert solucion.spread == pytest.approx(spread_total_teorico, abs=1e-3)
