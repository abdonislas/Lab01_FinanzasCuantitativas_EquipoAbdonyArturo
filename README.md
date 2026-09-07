# Lab01_Abdon_Arturo

## Integrantes

- Abdon Islas Leon
- Arturo Santillanes Llamas

## Descripción

Este proyecto implementa el modelo de market maker de Copeland y Galai: cotiza Bid y Ask frente a traders informados y de liquidez (no informados), maximiza la utilidad esperada por llegada y evalúa el resultado con simulaciones. El informado solo opera cuando le conviene. El código obtiene las cotizaciones óptimas, las compara con regímenes estrecho y amplio, valida el simulador contra la fórmula teórica, corre Monte Carlo (P&L, riesgo de pérdida e inventario) y genera las figuras del laboratorio.

## Instalación

Desde la raíz del repositorio:

```bash
python -m venv .venv
```

PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Mac / Linux:

```bash
source .venv/bin/activate
```

Luego instala las dependencias:

```bash
pip install -r requirements.txt
```

## Reproducir todos los resultados

Con el entorno virtual activado y desde la raíz del repositorio:

```bash
python main.py
```


**Semilla aleatoria:** `DEFAULT_SEED = 42`, definida en `src/simulation.py`. 
## Modelo

Utilidad esperada por trader que llega:

```
Pi(A,B) = pi_L * [ pi_LB(A - S0) * (A - S0) + pi_LS(S0 - B) * (S0 - B) ]
        - pi_I * [ int_A^inf (P - A) f(P) dP  +  int_0^B (B - P) f(P) dP ]
```

| Parámetro | Valor |
|---|---|
| S0 | 19.90 |
| P | Erlang(K=60, lambda=3), E[P] = 20.00, sd = 2.58 |
| pi_I / pi_L | 0.40 / 0.60 |
| pi_LB(s) = pi_LS(s) | max(0, 0.50 - 0.08 s) |

Supuestos: el market maker cotiza una unidad por lado, es neutral al riesgo y no gestiona inventario; el informado observa P sin ruido y solo opera cuando le conviene; la demanda no informada es simétrica y se trunca en cero.

## Resultados

### Cotizaciones óptimas

| Bid | Ask | Spread | Utilidad esperada por trade |
|---|---|---|---|
| 16.45 | 23.43 | 6.98 | 0.84 |

Descomposición: +0.9247 de ganancia frente a liquidez, -0.0539 de pérdida por el lado Ask, -0.0305 por el lado Bid.

Validación del simulador: con 500,000 llegadas y ejecución no forzada, el P&L medio simulado es 0.8364 contra 0.8403 teórico (error 0.0039). El simulador y la fórmula describen el mismo mecanismo.

### Simulación de 10,000 trades (ejecución forzada)

| Régimen | Bid | Ask | Spread | P&L total | P&L vs liquidez | P&L vs informados | Trades informados | Inv. final | Max abs inv |
|---|---|---|---|---|---|---|---|---|---|
| Óptimo | 16.45 | 23.43 | 6.98 | 19,925.23 | 20,821.74 | -896.51 | 713 | -105 | 148 |
| Estrecho | 19.75 | 20.05 | 0.30 | -6,887.78 | 894.90 | -7,782.68 | 3,843 | -51 | 117 |
| Amplio | 18.40 | 21.40 | 3.00 | 5,246.39 | 8,949.00 | -3,702.61 | 2,249 | 1 | 88 |

### Monte Carlo: 1,000 corridas de 1,000 trades

| Régimen | P&L promedio | Desv. estándar | Prob. de pérdida |
|---|---|---|---|
| Óptimo | 2,011.70 | 58.39 | 0.0% |
| Estrecho | -673.39 | 44.84 | 100.0% |
| Amplio | 542.74 | 44.83 | 0.0% |

### Sensibilidad a pi_I

| pi_I | Bid | Ask | Spread | Utilidad |
|---|---|---|---|---|
| 0.0 | 16.77 | 23.02 | 6.25 | 1.56 |
| 0.1 | 16.71 | 23.11 | 6.40 | 1.38 |
| 0.4 | 16.45 | 23.43 | 6.98 | 0.84 |
| 0.7 | 16.01 | 24.00 | 7.99 | 0.34 |

## Preguntas de análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread? Explíquelo con sus cifras del régimen estrecho.

En el régimen estrecho (19.75 / 20.05) el spread de 0.30 es menor que la desviación estándar de P (2.58), así que casi cualquier informado encuentra rentable operar: se ejecutaron 3,843 trades informados en 10,000 llegadas, contra 713 en el régimen óptimo. Cada uno de esos trades le cuesta al market maker la diferencia entre el valor verdadero y su cotización. La pérdida acumulada frente a informados fue de -7,782.68, mientras que la ganancia frente a liquidez fue de solo 894.90. El resultado neto es -6,887.78 en 10,000 trades y una probabilidad de pérdida del 100% en las 1,000 corridas de Monte Carlo. Con spread cero el market maker sería un seguro gratuito para el informado; el spread existe para que la ganancia frente a los no informados cubra esa pérdida.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread? Muéstrelo, no lo afirme.

Comparando los tres regímenes simulados, la pérdida esperada frente a informados cae conforme el spread se ensancha: -7,782.68 en el estrecho (spread 0.30), -3,702.61 en el amplio (spread 3.00), y -896.51 en el óptimo (spread 6.98). La caída no es proporcional al spread: de 0.30 a 3.00 (10 veces más ancho) la pérdida baja a menos de la mitad, pero de 3.00 a 6.98 (poco más del doble) la pérdida vuelve a caer a menos de una cuarta parte. Esto es porque cada unidad adicional de spread excluye a informados con señales cada vez más extremas, que bajo la distribución Erlang son cada vez menos probables: hay rendimientos marginales decrecientes en ampliar el spread como protección.

### 3. ¿Cuál régimen acumula el mayor desbalance de inventario y por qué? ¿A qué riesgo real lo expone eso, que el modelo no captura?

El régimen óptimo: inventario final de -105 y máximo absoluto de 148, contra 117 en el estrecho y 88 en el amplio. En las 1,000 corridas de Monte Carlo el inventario medio del óptimo es -7.32, consistente con la deriva teórica de -7.60 por cada 1,000 trades. El signo negativo viene de la asimetría de la Erlang: su cola derecha es más pesada que la izquierda, así que con cotizaciones casi simétricas alrededor de S0 hay más informados comprando (P > 23.43) que vendiendo (P < 16.45), y el market maker termina corto. Los traders de liquidez son 50/50 y no aportan deriva sistemática.

El riesgo real que esto expone, y que el modelo no captura, es el riesgo de inventario: una posición corta de 105 unidades valuada a S0 = 19.90 son 2,090 de exposición direccional. El modelo asume neutralidad al riesgo y no penaliza la posición acumulada, así que nunca ajusta las cotizaciones para descargarla. Un market maker real sesgaría el Bid y el Ask hacia abajo para atraer vendedores, aceptando menos utilidad por trade a cambio de reducir la exposición.

### 4. ¿Cómo se comporta el spread óptimo al variar pi_I? ¿Coincide con la teoría?

Crece de forma monótona: 6.25 con pi_I = 0, 6.40 con 0.1, 6.98 con 0.4 y 7.99 con 0.7. La utilidad esperada cae de 1.56 a 0.34 en el mismo rango. Esto coincide con la teoría: sin informados el market maker es un monopolista frente a la demanda lineal, y el máximo de (0.50 - 0.08 s) * s está en s* = 0.50 / (2 * 0.08) = 3.125 por lado, es decir un spread total de 0.50 / 0.08 = 6.25, que el optimizador reproduce con error 0.0000. Conforme sube pi_I, la pérdida esperada por selección adversa pesa más en la condición de primer orden y el óptimo se desplaza hacia spreads más anchos.

### 5. Mencione tres limitaciones de este modelo para un market maker real.

1. **No hay gestión de inventario ni aversión al riesgo.** El market maker real ajusta sus cotizaciones según su posición acumulada. Aquí las cotizaciones son fijas y el inventario deriva sin control, como muestra el régimen óptimo.
2. **El precio de referencia no se actualiza.** Cada trade de un informado revela información sobre P, pero S0 permanece en 19.90 durante los 10,000 trades. En un modelo secuencial el market maker actualizaría S0 de forma bayesiana tras cada orden y el spread se cerraría conforme aprende.
3. **La ejecución es por trade, no por unidad de tiempo.** No hay tasa de llegada, ni competencia entre market makers, ni costo de oportunidad del capital inmovilizado, factores que en la práctica limitan el spread mucho antes que la selección adversa.

## Advertencia de interpretación

La simulación fuerza la ejecución de un trade en cada iteración, por lo que todos los resultados de las secciones de simulación y Monte Carlo representan rentabilidad por trade y no rentabilidad por unidad de tiempo. Bajo esta métrica un spread muy amplio, que en un mercado real casi nunca se ejecutaría, sale artificialmente favorecido. Esto explica por qué el régimen óptimo del modelo (spread 6.98) es más ancho que el régimen etiquetado como amplio (3.00) y aun así lo supera en P&L: con ejecución forzada, cada trader de liquidez paga el spread completo sin importar su tamaño. La sección 3 de `main.py` corre el simulador sin forzar ejecución (respetando pi_LB y pi_LS) y confirma que en ese caso el P&L medio por llegada converge a la utilidad teórica de 0.84, no al 1.99 que reporta la simulación forzada.
