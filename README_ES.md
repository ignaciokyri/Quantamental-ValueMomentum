# Estrategia Cuantamental Value-Momentum

Una estrategia de renta variable que combina la **selección fundamental de acciones** (cartera del fondo COBAS Selección FI) con un filtro de **momentum 12-1 skip-month**. Desarrollada íntegramente en Python con un pipeline completo de validación estadística.

---

## ¿Qué hace?

[COBAS Selección FI](https://www.cobasam.com/productos/inversion-libre/cobas_seleccion/) es un fondo de valor gestionado por Francisco García Paramés, uno de los inversores a largo plazo más reconocidos de Europa. El fondo mantiene entre ~40 y 60 empresas pequeñas y medianas internacionales de más de 15 países, divulgadas públicamente cada seis meses a través de los informes regulatorios del supervisor español (CNMV).

Esta estrategia utiliza esas divulgaciones como un **universo de valor pre-filtrado**, y lo filtra adicionalmente con momentum 12-1 — conservando únicamente el top 15% de los valores con mayor tendencia de precio en los 11 meses anteriores (saltando el mes más reciente para evitar la reversión a corto plazo).

```
Informes semestrales CNMV
        │
        ▼
  Extracción PDF (pdfplumber)  ──►  Lista de ISINs por semestre
        │
        ▼
  ISIN → ticker de Yahoo Finance
        │
        ▼
  Descarga de precios (yfinance)
        │
        ▼
  Señal de momentum 12-1
        │
        ▼
  Selección top 15% (ponderación igual)
        │
        ▼
  Rebalanceo mensual  ──►  Curva de equity, drawdown, métricas
```
---

## Período del backtest

| | |
|---|---|
| Fuente del universo | COBAS Selección FI (informes CNMV) |
| Semestres cubiertos | 2017 S1 → 2025 S2 (16 semestres) |
| Ventana del backtest | ~2017 – presente |
| Rebalanceo | Mensual |
| Tamaño de cartera | Top 15% por momentum (~6–9 posiciones) |
| Ponderación | Igual peso |
| Benchmark | S&P 500 (SPY) |
---

## Outputs principales

| Output | Descripción |
|--------|-------------|
| `outputs/backtest_dashboard.png` | Dashboard de 4 paneles: curva de equity, drawdown, barras de retorno mensual, heatmap mensual |
| `outputs/backtest_equity_curves.png` | Estrategia vs COBAS Selección FI vs S&P 500 |
| `outputs/grid_heatmaps.png` | CAGR y Sharpe en 100 combinaciones de parámetros |
| `outputs/grid_equity_top5.png` | Top 5 y bottom 5 combinaciones por CAGR |
| `outputs/borda_heatmap.png` | Consistencia Borda Count: qué combinación gana más frecuentemente mes a mes |
| `outputs/borda_acumulado.png` | Puntos Borda acumulados en el tiempo — muestra si el ganador es consistente a lo largo del período |
| `outputs/hypothesis_pvalues.png` | Heatmap de p-valores corregidos por BHY |
| `outputs/costs.png` | Equity neta bajo IRPF (particular) vs comisiones vs sin costes |

> **IRPF** (*Impuesto sobre la Renta de las Personas Físicas*) es el impuesto español sobre la renta que grava las ganancias de capital. Los tipos son progresivos: 19% hasta 6.000€, 21% hasta 50.000€, 23% hasta 200.000€, 26% hasta 300.000€ y 28% a partir de ahí. Las ganancias solo tributan en el momento de la venta.

---

## Validación estadística

La configuración 1M/15% fue validada mediante un pipeline de tres etapas para descartar sobreajuste de parámetros:

**1. Grid Search (100 combinaciones)**
Se probaron 10 frecuencias de rebalanceo (1S a 10S) × 10 tamaños de cartera (5% a 50%). Genera heatmaps de CAGR, Sharpe y DrawDown máximo.

**2. Borda Count (test de consistencia)**
Para cada mes natural, rankea las 100 combinaciones por rentabilidad mensual. Acumula puntos a lo largo de toda la historia. La configuración que consistentemente ocupa posiciones altas —independientemente del retorno total— obtiene más puntos.

**3. Test de hipótesis (significación estadística)**
Aplicado a 110 combinaciones (grid 10S + 1M explícito):
- **Block bootstrap** (bloque = 3 meses, N = 3.000 muestras) — preserva la autocorrelación típica del momentum mensual
- **Test pareado de Sharpe** — H₀: Sharpe(1M/15%) ≤ Sharpe(rival); p-valor = fracción de muestras bootstrap donde el objetivo no supera al rival
- **Corrección Benjamini-Yekutieli** — controla la tasa de falsos descubrimientos (FDR) bajo dependencia arbitraria en 109 tests simultáneos
- **White's Reality Check** — p-valor global que corrige el hecho de que 1M/15% fue elegida *después* de ver todos los resultados

---

## Estructura del proyecto

```
├── main/
│   ├── backtest.py                    # Estrategia principal: backtest + dashboard
│   ├── ISIN_extracter.py              # Extracción PDF → ISIN (pdfplumber)
│   ├── ISIN_to_ticker_converter.py    # Mapeo ISIN → ticker de Yahoo Finance
│   └── main.py                        # Ejecuta el pipeline completo
├── tests/
│   ├── test_grid.py                   # Grid search: 100 combinaciones frecuencia × top-N
│   ├── test_borda.py                  # Ranking de consistencia mensual Borda Count
│   ├── test_hypothesis.py             # Block bootstrap + BHY + White's RC
│   └── test_costs.py                  # Escenarios after-tax: IRPF vs sin costes
├── outputs/
│   ├── tickers.txt                    # Universo procesado (ticker, empresa, año, semestre)
│   └── *.png                          # Gráficos generados
└── docs/
    ├── USAGE.md                       # Guía de obtención de datos (inglés)
    └── USAGE_ES.md                    # Guía de obtención de datos (español)
```

---

## Instalación

```bash
pip install -r requirements.txt
```

**Datos necesarios** (no incluidos en este repositorio por derechos de autor): Consultar `docs/USAGE`

---

## Uso

```bash
# Pipeline completo (recomendado)
python main/main.py

# Estrategia principal + dashboard
python main/backtest.py

# Grid search de parámetros (100 combinaciones, ~10 min en paralelo)
python tests/test_grid.py

# Análisis de consistencia Borda Count
python tests/test_borda.py

# Test de hipótesis estadístico
python tests/test_hypothesis.py

# Análisis de escenarios después de costes
python tests/test_costs.py
```

> En Windows, ejecuta `$env:PYTHONIOENCODING="utf-8"` antes de correr los scripts para evitar problemas de codificación con caracteres especiales.

---

## Limitaciones conocidas

- **Riesgo de divisa:** Los precios se descargan en monedas locales (USD, GBP, EUR, KRW…). No se aplica conversión de divisa. La curva de equity de la estrategia mezcla divisas, mientras que el VL de COBAS está denominado en EUR.
- **Bruto de costes:** El backtest principal no incluye costes de transacción. `test_costs.py` modela las comisiones de IBKR (0,05% por operación, mínimo 3,50€).
- **Sesgo de supervivencia (datos):** Solo se utilizan acciones con histórico disponible en Yahoo Finance. Los valores deslistados sin datos se excluyen silenciosamente.

---

## Dependencias

Ver `requirements.txt`. Librerías principales: `pandas`, `numpy`, `yfinance`, `matplotlib`, `scipy`, `pdfplumber`, `openpyxl`.
