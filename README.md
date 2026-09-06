# Lab01_Abdon_Arturo

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

Ese comando imprime en consola parámetros, cotizaciones óptimas, validación teórica, simulaciones por régimen, Monte Carlo y sensibilidad a \(\pi_I\), y guarda las figuras en `docs/figures/`. La semilla es `42`, así que la corrida es reproducible.
