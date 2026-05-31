"""
test_costs.py
Estrategia 12-1 (1M / Top 15%) sometida a dos regimenes fiscales espanoles:
  Escenario 1 — Inversor Particular : IRPF del ahorro (tramos progresivos)
  Escenario 2 — Sociedad Limitada   : Impuesto de Sociedades (25% tipo general)
Benchmark     — MSCI EAFE Small-Cap (SCZ) : sin coste fiscal, base 50.000 EUR

Friccion operativa: 0.05% comision IBKR + minimo 3.5 EUR por orden.
Look-ahead bias corregido (mismo criterio que backtest.py).
"""

import importlib.util, sys, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Importar backtest.py (reutilizar funciones ya corregidas) ────────────────
BASE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bt", BASE / "main" / "backtest.py")
bt   = importlib.util.module_from_spec(spec)
sys.stdout = io.StringIO()
try:
    spec.loader.exec_module(bt)
finally:
    sys.stdout = sys.__stdout__

OUTPUTS_DIR = BASE / "outputs"

# ── Parametros de simulacion ──────────────────────────────────────────────────
CAPITAL_INICIAL   = 50_000.0
TASA_COMISION     = 0.0005   # IBKR tarifa fija: 0.05% del valor de la operacion
MIN_COMISION_EUR  = 3.5      # IBKR minimo por orden (3€ Europa / 4€ otros → media 3.5€)
TOP_PCT           = bt.TOP_PCT  # importado de backtest.py (actualmente 0.15)

# ── Tramos fiscales ───────────────────────────────────────────────────────────

def calcular_irpf_ahorro(base: float) -> float:
    """
    Escala progresiva IRPF del ahorro (Espana, vigente 2024).
    Solo se llama si base > 0 (ya aplicadas compensaciones de perdidas).
    """
    impuesto = 0.0
    tramos = [
        (  6_000, 0.19),
        ( 44_000, 0.21),   # tramo 6.000 – 50.000
        (150_000, 0.23),   # tramo 50.000 – 200.000
        (100_000, 0.26),   # tramo 200.000 – 300.000
        (float("inf"), 0.28),
    ]
    resto = base
    for limite, tipo in tramos:
        tramo     = min(resto, limite)
        impuesto += tramo * tipo
        resto    -= tramo
        if resto <= 0:
            break
    return impuesto


def calcular_is(base: float) -> float:
    """Impuesto de Sociedades: tipo general 25 %."""
    return max(0.0, base * 0.25)


# ── Rebalanceo con seguimiento de base de coste ───────────────────────────────

def rebalancear(posiciones: dict, nueva_cartera: list, tasa: float,
                min_com: float = 0.0) -> tuple:
    """
    Rebalancea la cartera a igual ponderacion aplicando comisiones IBKR:
      max(valor_operacion * tasa, min_com) por orden.

    Cada posicion se representa como {'valor': float, 'coste': float}.
    - valor : precio de mercado actual del bloque
    - coste : base de coste acumulada (lo que se pago en total)

    Devuelve (nuevas_posiciones, pnl_realizado_neto, comisiones_pagadas).
    """
    def _com(val: float) -> float:
        return max(val * tasa, min_com) if val > 0 else 0.0

    set_v  = set(posiciones.keys())
    set_n  = set(nueva_cartera)
    n      = len(nueva_cartera)
    vt     = sum(p["valor"] for p in posiciones.values())

    # 1) Estimacion de comisiones para calcular el valor neto disponible
    tgt_prev = vt / n if n > 0 else 0.0
    com  = sum(_com(posiciones[t]["valor"])                    for t in set_v - set_n)
    com += sum(_com(tgt_prev)                                   for _ in set_n - set_v)
    com += sum(_com(abs(posiciones[t]["valor"] - tgt_prev))     for t in set_n & set_v)

    valor_neto = vt - com
    tgt        = valor_neto / n if n > 0 else 0.0

    nuevas = {}
    pnl    = 0.0

    # 2) Ventas totales (posiciones que salen del portfolio)
    for t in set_v - set_n:
        p    = posiciones[t]
        pnl += p["valor"] - p["coste"]          # ganancia/perdida realizada

    # 3) Posiciones que permanecen: ajustar peso
    for t in set_n & set_v:
        p    = posiciones[t]
        diff = tgt - p["valor"]

        if diff < 0:                             # reduccion de posicion (venta parcial)
            frac = abs(diff) / p["valor"] if p["valor"] > 0 else 0.0
            pnl += frac * (p["valor"] - p["coste"])
            nuevas[t] = {
                "valor": tgt,
                "coste": p["coste"] * (1.0 - frac),
            }
        else:                                    # aumento de posicion (compra adicional)
            nuevas[t] = {
                "valor": tgt,
                "coste": p["coste"] + diff,
            }

    # 4) Entradas nuevas
    for t in set_n - set_v:
        nuevas[t] = {"valor": tgt, "coste": tgt}

    return nuevas, pnl, com


# ── Backtest principal con fiscalidad ─────────────────────────────────────────

def ejecutar_fiscal(
    universo: dict,
    precios:  pd.DataFrame,
    regimen:  str,           # 'irpf' | 'is' | 'sin_impuestos'
    tasa:     float = TASA_COMISION,
    min_com:  float = MIN_COMISION_EUR,
) -> pd.Series:
    """
    Simula la estrategia con comisiones e impuesto liquidado al final de cada
    ejercicio fiscal (ultimo dia habíl de diciembre).

    IRPF: perdidas compensables hasta 4 ejercicios anteriores.
    IS  : base imponible negativa compensable sin limite temporal.
    """
    periodos   = sorted(universo.keys())
    ano_ini, si = periodos[0]
    ano_fin, sf = periodos[-1]
    primer_dia  = pd.Timestamp(ano_ini, 1 if si == 1 else 7, 1)
    ultimo_dia  = pd.Timestamp(ano_fin, 6 if sf == 1 else 12, 31)
    inicios_mes = pd.date_range(start=primer_dia, end=ultimo_dia, freq="MS")

    pos:  dict[str, dict] = {}
    hist: dict             = {}

    pnl_anno = 0.0
    # IRPF: lista de (ano_perdida, importe_negativo) — max 4 anos de offset
    perdidas_irpf: list[tuple[int, float]] = []
    # IS: base imponible negativa acumulada (offset ilimitado)
    base_neg_is = 0.0

    print(f"\n{'='*68}")
    print(f"  REGIMEN: {regimen.upper():<8}  |  "
          f"comision={TASA_COMISION:.2%}  |  top={TOP_PCT:.0%}  |  capital={CAPITAL_INICIAL:,.0f}€")
    print(f"{'='*68}")
    print(f"  {'Año':<6} {'PnL realiz.':>14} {'Base impon.':>14} "
          f"{'Impuesto':>12} {'Equity final':>14}")
    print(f"  {'-'*62}")

    for mes_ini_cal in inicios_mes:
        ano = mes_ini_cal.year
        sem = 1 if mes_ini_cal.month <= 6 else 2

        # Look-ahead fix: usar ultimo semestre ya publicado
        clave = (ano - 1, 2) if mes_ini_cal.month <= 7 else (ano, 1)
        if clave not in universo:
            continue

        fin_mes_cal = mes_ini_cal + pd.offsets.MonthEnd(0)
        dias_mes    = precios.index[
            (precios.index >= mes_ini_cal) & (precios.index <= fin_mes_cal)
        ]
        if dias_mes.empty:
            continue

        f_ini, f_fin = dias_mes[0], dias_mes[-1]

        tickers_per = [t.upper() for t in universo[clave]]
        momentos    = bt.calcular_momento_12_1(precios, tickers_per, f_ini)
        cartera     = bt.seleccionar_top(momentos, TOP_PCT)
        if not cartera:
            continue

        # Primer mes: desplegar capital inicial (N ordenes de compra)
        if not pos:
            n   = len(cartera)
            tgt = CAPITAL_INICIAL / n
            com_entrada = sum(max(tgt * tasa, min_com) for _ in cartera)
            capital_neto = CAPITAL_INICIAL - com_entrada
            tgt = capital_neto / n
            pos = {t: {"valor": tgt, "coste": tgt} for t in cartera}
        else:
            pos, pnl_mes, _ = rebalancear(pos, cartera, tasa, min_com)
            pnl_anno += pnl_mes

        # Retornos diarios sobre ventana minima (optimizacion O(periodo))
        cartera_ok = [t for t in cartera if t in precios.columns]
        if not cartera_ok:
            continue

        idx_i = precios.index.searchsorted(f_ini)
        idx_f = precios.index.searchsorted(f_fin, side="right")
        vc    = precios.iloc[max(0, idx_i - 1):idx_f][cartera_ok]
        rets  = vc.pct_change().loc[f_ini:f_fin].fillna(0.0)

        for fecha, r in rets.iterrows():
            for t in pos:
                if t in r.index:
                    pos[t]["valor"] *= (1.0 + float(r[t]))
            hist[fecha] = sum(p["valor"] for p in pos.values())

        # ── Liquidacion fiscal: fin del ejercicio (ultimo dia habíl diciembre) ──
        if f_fin.month == 12:
            ano_fiscal = f_fin.year
            vt         = sum(p["valor"] for p in pos.values())

            if regimen == "sin_impuestos":
                impuesto = 0.0
                base     = pnl_anno

            elif regimen == "irpf":
                # Expirar perdidas con mas de 4 anos de antiguedad
                perdidas_irpf = [
                    (y, imp) for y, imp in perdidas_irpf
                    if y >= ano_fiscal - 4
                ]
                # Compensar base imponible con perdidas anteriores
                base = pnl_anno
                nuevas_p: list[tuple[int, float]] = []
                for y, imp in perdidas_irpf:
                    if base > 0 and imp < 0:
                        usar  = min(base, -imp)
                        base -= usar
                        resto = imp + usar
                        if resto < 0:
                            nuevas_p.append((y, resto))
                    else:
                        nuevas_p.append((y, imp))
                perdidas_irpf = nuevas_p
                if pnl_anno < 0:
                    perdidas_irpf.append((ano_fiscal, pnl_anno))
                impuesto = calcular_irpf_ahorro(base) if base > 0 else 0.0

            else:  # IS
                base = pnl_anno + base_neg_is
                if base > 0:
                    impuesto = calcular_is(base)
                    base_neg_is = 0.0
                else:
                    impuesto    = 0.0
                    base_neg_is = base

            # Deducir impuesto de la cartera (reduccion proporcional)
            if impuesto > 0.0 and vt > 0.0:
                factor = max(0.0, (vt - impuesto) / vt)
                for t in pos:
                    pos[t]["valor"] *= factor
                    pos[t]["coste"] *= factor
                hist[f_fin] = sum(p["valor"] for p in pos.values())

            if regimen != "sin_impuestos":
                equity_fin = hist.get(f_fin, vt)
                print(
                    f"  {ano_fiscal:<6} "
                    f"{pnl_anno:>+14,.0f}€ "
                    f"{base:>+14,.0f}€ "
                    f"{impuesto:>11,.0f}€ "
                    f"{equity_fin:>13,.0f}€"
                )

            pnl_anno = 0.0

    if not hist:
        return pd.Series(dtype=float)

    eq = pd.Series(hist, name=regimen)
    print(f"\n  Capital final ({regimen.upper()}): {float(eq.iloc[-1]):,.0f} €")
    return eq


# ── Metricas ─────────────────────────────────────────────────────────────────

def metricas(eq: pd.Series) -> dict:
    if eq.empty:
        return {k: np.nan for k in ["ret", "cagr", "vol", "sharpe", "maxdd", "capital_final"]}
    eq  = eq.astype(float).dropna()
    v0, v1  = float(eq.iloc[0]), float(eq.iloc[-1])
    n_anos  = (eq.index[-1] - eq.index[0]).days / 365.25
    rd      = eq.pct_change().dropna()
    vol     = float(rd.std()) * np.sqrt(252)
    cagr    = (v1 / v0) ** (1 / n_anos) - 1 if n_anos > 0 else np.nan
    sharpe  = (float(rd.mean()) * 252) / vol if vol > 0 else np.nan
    maxdd   = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {
        "ret":           v1 / v0 - 1,
        "cagr":          cagr,
        "vol":           vol,
        "sharpe":        sharpe,
        "maxdd":         maxdd,
        "capital_final": v1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCION
# ═══════════════════════════════════════════════════════════════════════════════

print("Cargando universo y precios...")
universo = bt.cargar_universo(bt.RUTA_TICKERS)
precios  = bt.descargar_precios(universo)
print("Listo.")

eq_bruto   = ejecutar_fiscal(universo, precios, "sin_impuestos", tasa=0.0, min_com=0.0)
eq_sin_imp = ejecutar_fiscal(universo, precios, "sin_impuestos")
eq_irpf    = ejecutar_fiscal(universo, precios, "irpf")

# Benchmark normalizado a capital inicial
print("\nDescargando MSCI EAFE Small-Cap (SCZ)...")
f0 = eq_irpf.index[0]
f1 = eq_irpf.index[-1]
bm_raw = bt.descargar_benchmark(f0, f1)
bm_eur = (bm_raw / bm_raw.iloc[0] * CAPITAL_INICIAL
          if not bm_raw.empty else pd.Series(dtype=float))
bm_eur.name = "benchmark"

# ── Tabla comparativa ─────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print(f"{'COMPARATIVA FINAL':^74}")
print("=" * 74)
print(f"  {'Escenario':<36} {'Capital Final':>13} {'CAGR':>7} "
      f"{'Sharpe':>7} {'Max DD':>8} {'Volat.':>7}")
print("  " + "-" * 70)

escenarios = [
    (bm_eur,     "MSCI EAFE Small-Cap (SCZ) — sin gastos"),
    (eq_bruto,   "Estrategia — Sin gastos"),
    (eq_sin_imp, "Estrategia — Con comisiones (sin impuestos)"),
    (eq_irpf,    "Estrategia — Comisiones + impuestos (IRPF)"),
]
for eq, nombre in escenarios:
    m = metricas(eq)
    print(
        f"  {nombre:<48} "
        f"{m['capital_final']:>12,.0f}€ "
        f"{m['cagr']:>7.2%} "
        f"{m['sharpe']:>7.2f} "
        f"{m['maxdd']:>8.2%}"
    )
print("=" * 74)

# ── Desglose de friccion ───────────────────────────────────────────────────────
m_bruto = metricas(eq_bruto)
m_sin   = metricas(eq_sin_imp)
m_irpf  = metricas(eq_irpf)
print(f"\n  DESGLOSE DE FRICCION (en puntos de CAGR):")
print(f"  Coste comisiones    : {m_sin['cagr'] - m_bruto['cagr']:+.2%}  "
      f"({m_bruto['capital_final'] - m_sin['capital_final']:,.0f}€ drenados en comisiones)")
print(f"  Impuesto IRPF extra : {m_irpf['cagr'] - m_sin['cagr']:+.2%}  "
      f"({m_sin['capital_final'] - m_irpf['capital_final']:,.0f}€ drenados en impuestos)")
print("=" * 74)

# ── Grafica comparativa ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

for eq, color, lw, ls, label in [
    (bm_eur,     "#2980b9", 1.5, "--", "MSCI EAFE Small-Cap (SCZ) — sin gastos"),
    (eq_bruto,   "#27ae60", 2.0, "-",  "Estrategia — Sin gastos"),
    (eq_sin_imp, "#8e44ad", 2.0, "-",  "Estrategia — Con comisiones (sin impuestos)"),
    (eq_irpf,    "#c0392b", 2.0, "-",  "Estrategia — Comisiones + impuestos (IRPF)"),
]:
    if not eq.empty:
        ax.plot(eq.index, eq.values,
                color=color, linewidth=lw, linestyle=ls, label=label)

ax.axhline(CAPITAL_INICIAL, color="gray", linewidth=0.8, linestyle=":",
           label=f"Capital inicial ({CAPITAL_INICIAL:,.0f}€)")
ax.set_title(
    f"Estrategia 12-1 (1M / Top {int(TOP_PCT*100)}%) — Impacto Fiscal: Sin gastos vs. Comisiones vs. IRPF vs. SCZ",
    fontsize=13, fontweight="bold",
)
ax.set_ylabel("Capital (EUR)", fontsize=11)
ax.set_xlabel("Fecha", fontsize=11)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()

ruta = OUTPUTS_DIR / "costs.png"
plt.savefig(ruta, dpi=150)
plt.show()
print(f"\nGrafica guardada en: {ruta}")
