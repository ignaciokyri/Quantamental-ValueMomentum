"""
Estrategia 12-1: Universo Value (Parames/COBAS) filtrado por Momentum 12-1
Configuracion optima: Rebalanceo Mensual, Top 15%

Fuentes de datos:
  - outputs/tickers_cobas.txt -> universo elegible por semestre (Yahoo tickers)
  - seleccion-diario.xlsx -> NAV diario del fondo COBAS Seleccion FI
  - Yahoo Finance (yfinance)  -> precios de cierre ajustados de cada accion
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ─── Configuracion ────────────────────────────────────────────────────────────
# Fix #6: rutas dinamicas, sin hardcodear usuario ni maquina
BASE_DIR     = Path(__file__).resolve().parent
OUTPUTS_DIR  = BASE_DIR.parent / "outputs"
RUTA_TICKERS = OUTPUTS_DIR / "tickers.txt"
RUTA_EXCEL   = BASE_DIR.parent / "data" / "seleccion-diario.xlsx"

TOP_PCT = 0.15   # top 15% por momentum (optimo Borda Count mensual)

COLOR_ESTRATEGIA = "#c0392b"   # rojo
COLOR_COBAS      = "#27ae60"   # verde
COLOR_BM         = "#2980b9"   # azul


# ===============================================================================
# 1. CARGA DE DATOS
# ===============================================================================

def cargar_universo(ruta: Path) -> dict[tuple[int, int], list[str]]:
    """
    Lee tickers_cobas.txt y devuelve {(anno, semestre): [ticker, ...]} .
    Parsea linea a linea para tolerar comas en nombres de empresa.
    Formato: "Ticker, Nombre (puede tener comas), Anno, Semestre N"
    """
    registros = []
    with open(ruta, encoding="utf-8") as f:
        for num, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            partes = [p.strip() for p in linea.split(",")]
            if len(partes) < 4:
                print(f"  [tickers_cobas.txt] Linea {num} ignorada: {linea}")
                continue
            ticker  = partes[0].upper()
            ano_str = partes[-2]
            sem_str = partes[-1]
            try:
                ano = int(ano_str)
            except ValueError:
                continue
            m = re.search(r"(\d)", sem_str)
            if not m:
                continue
            registros.append((ticker, ano, int(m.group(1))))

    universo: dict[tuple[int, int], list[str]] = {}
    for ticker, ano, sem in registros:
        universo.setdefault((ano, sem), []).append(ticker)
    return universo


def cargar_nav_fondo(ruta: Path) -> pd.Series:
    """
    Lee seleccion-diario.xlsx (columnas: Fecha, Liquidativo).
    Devuelve pd.Series normalizada a base 100.
    """
    df = pd.read_excel(ruta, engine="openpyxl", parse_dates=["Fecha"])
    df.columns = [c.strip() for c in df.columns]
    col_fecha = next((c for c in df.columns if "fecha" in c.lower()), df.columns[0])
    # Fix #7: buscar por nombre "liquidat" antes de caer al primer no-fecha
    col_nav = next(
        (c for c in df.columns if "liquidat" in c.lower()),
        next((c for c in df.columns if c.lower() != col_fecha.lower()), df.columns[1]),
    )
    df = df[[col_fecha, col_nav]].dropna()
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df = df.dropna(subset=[col_fecha]).set_index(col_fecha).sort_index()
    nav = df[col_nav].astype(float)
    nav = nav / nav.iloc[0] * 100
    nav.name = "COBAS Seleccion FI"
    print(f"  Fondo: {len(nav)} obs  ({nav.index[0].date()} -> {nav.index[-1].date()})")
    return nav


def descargar_precios(universo: dict[tuple[int, int], list[str]]) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados desde Yahoo Finance.
    Ventana: 13 meses antes del primer semestre hasta fin del ultimo.
    """
    periodos  = sorted(universo.keys())
    todos_tkr = sorted({t for tickers in universo.values() for t in tickers})
    ano_ini, sem_ini = periodos[0]
    ano_fin, sem_fin = periodos[-1]
    fecha_dl_ini = pd.Timestamp(ano_ini, 1 if sem_ini == 1 else 7, 1) - pd.DateOffset(months=13)
    fecha_dl_fin = pd.Timestamp(ano_fin, 6 if sem_fin == 1 else 12, 30 if sem_fin == 1 else 31)

    print(f"  Descargando {len(todos_tkr)} tickers ({fecha_dl_ini.date()} -> {fecha_dl_fin.date()}) ...")
    raw = yf.download(
        todos_tkr,
        start=fecha_dl_ini,
        end=fecha_dl_fin + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Fix #3/#4: extraccion robusta ante MultiIndex y ticker unico
    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    elif "Close" in raw.columns:
        precios = raw[["Close"]]
    else:
        precios = raw

    # yfinance devuelve Series cuando hay un solo ticker → forzar DataFrame
    if isinstance(precios, pd.Series):
        precios = precios.to_frame()

    # MultiIndex residual con tuplas vacias, ej. ('AAPL', '') → 'AAPL'
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)

    precios.columns = [str(c).strip().upper() for c in precios.columns]
    precios = precios.sort_index()
    print(f"  Descargados: {len(precios.columns)} tickers ({precios.index[0].date()} -> {precios.index[-1].date()})")
    return precios


def descargar_benchmark(fecha_inicio: pd.Timestamp, fecha_fin: pd.Timestamp) -> pd.Series:
    """Descarga SCZ (MSCI EAFE Small-Cap) y devuelve pd.Series base 100.

    SCZ es el benchmark más apropiado para COBAS: cubre small/mid caps de
    mercados desarrollados ex-US (Europa, Japón, Australia…), cotiza en USD
    en NYSE y tiene historial desde 2007.
    """
    raw = yf.download("SCZ", start=fecha_inicio, end=fecha_fin + pd.Timedelta(days=1),
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    elif "Close" in raw.columns:
        precios = raw["Close"]
    else:
        precios = raw.iloc[:, 0]

    if isinstance(precios, pd.DataFrame):
        precios = precios.squeeze()

    precios = precios.astype(float).dropna().sort_index()
    bm = precios / precios.iloc[0] * 100
    bm.name = "MSCI EAFE Small-Cap (SCZ)"
    print(f"  SCZ: {bm.index[0].date()} -> {bm.index[-1].date()}")
    return bm


# ===============================================================================
# 2. CALCULO DE MOMENTUM 12-1
# ===============================================================================

def _ultimo_precio_antes_de(serie: pd.Series, fecha: pd.Timestamp) -> float | None:
    datos = serie.loc[serie.index <= fecha].dropna()
    return float(datos.iloc[-1]) if not datos.empty else None


def calcular_momento_12_1(
    precios: pd.DataFrame,
    tickers: list[str],
    fecha_rebalan: pd.Timestamp,
) -> pd.Series:
    """
    Retorno P(T-1mes) / P(T-12meses) - 1 para cada ticker.
    El desfase de 1 mes excluye el mes mas reciente (mitiga reversion a corto).
    """
    t_fin    = fecha_rebalan - pd.DateOffset(months=1)
    t_inicio = fecha_rebalan - pd.DateOffset(months=12)
    momentos = {}
    for tkr in tickers:
        col = tkr.upper()
        if col not in precios.columns:
            continue
        p_fin    = _ultimo_precio_antes_de(precios[col], t_fin)
        p_inicio = _ultimo_precio_antes_de(precios[col], t_inicio)
        if p_fin is None or p_inicio is None or p_inicio == 0:
            continue
        momentos[col] = p_fin / p_inicio - 1
    return pd.Series(momentos, name="momentum_12_1")


def seleccionar_top(momentos: pd.Series, top_pct: float = TOP_PCT) -> list[str]:
    """
    Fix #2: funcion pura extraida para ser importable por los tests.
    Devuelve los tickers con mayor momentum (top top_pct%).
    Retorna lista vacia si momentos esta vacio.
    """
    if momentos.empty:
        return []
    n = max(1, int(np.ceil(len(momentos) * top_pct)))
    return momentos.nlargest(n).index.tolist()


# ===============================================================================
# 3. BACKTEST PARAMETRICO (frecuencia y top% configurables)
# ===============================================================================

def backtest_parametrico(
    universo: dict,
    precios: pd.DataFrame,
    freq: str,
    pct: float,
) -> pd.Series:
    """Backtest con frecuencia y top% arbitrarios. Base 100. Usado por los tests."""
    periodos  = sorted(universo.keys())
    ano_ini, sem_ini = periodos[0]
    ano_fin, sem_fin = periodos[-1]
    primer_dia = pd.Timestamp(ano_ini, 1 if sem_ini == 1 else 7, 1)
    ultimo_dia = pd.Timestamp(ano_fin, 6 if sem_fin == 1 else 12, 30 if sem_fin == 1 else 31)

    fechas = pd.date_range(start=primer_dia, end=ultimo_dia, freq=freq)
    tramos, valor = [], 100.0

    for i, f_ini_cal in enumerate(fechas):
        f_fin_cal = fechas[i + 1] - pd.Timedelta(days=1) if i + 1 < len(fechas) else ultimo_dia
        dias = precios.index[(precios.index >= f_ini_cal) & (precios.index <= f_fin_cal)]
        if dias.empty:
            continue
        f_ini, f_fin = dias[0], dias[-1]
        clave = (f_ini_cal.year - 1, 2) if f_ini_cal.month <= 7 else (f_ini_cal.year, 1)
        if clave not in universo:
            continue
        tickers  = [t.upper() for t in universo[clave]]
        momentos = calcular_momento_12_1(precios, tickers, f_ini)
        cartera  = seleccionar_top(momentos, pct)
        if not cartera:
            continue
        idx_ini = precios.index.searchsorted(f_ini)
        idx_fin = precios.index.searchsorted(f_fin, side="right")
        ancla   = max(0, idx_ini - 1)
        ventana = precios.iloc[ancla:idx_fin][cartera]
        ret = ventana.pct_change().loc[f_ini:f_fin].mean(axis=1).fillna(0.0)
        eq  = valor * (1 + ret).cumprod()
        tramos.append(eq)
        valor = float(eq.iloc[-1])

    if not tramos:
        return pd.Series(dtype=float)
    eq_total = pd.concat(tramos)
    return eq_total[~eq_total.index.duplicated(keep="last")]


# ===============================================================================
# 4. BACKTEST MENSUAL (1M / Top 15%)
# ===============================================================================

def ejecutar_backtest_mensual(
    universo: dict[tuple[int, int], list[str]],
    precios: pd.DataFrame,
) -> pd.Series:
    """
    Backtest con rebalanceo MENSUAL y seleccion Top TOP_PCT%.

    Cada mes:
      1. Universo = ultimo informe CNMV publicado (semestre anterior).
         El informe S1 (ene-jun) del anno Y se publica ~ago-Y, por lo
         que en ene-jun se usan los tickers de S2 del anno Y-1.
         En jul-dic se usan los tickers de S1 del anno Y (publicado ~ago-Y).
         Esto elimina el look-ahead bias estructural.
      2. Momentum 12-1 en el primer dia de negociacion del mes.
      3. Cartera = top TOP_PCT% por momentum, equiponderada.
      4. pct_change() sobre ventana minima (1 dia de ancla) para capturar
         el retorno real del primer dia sin procesar todo el historico.
      5. Equity encadenada mes a mes, base 100.
    """
    periodos = sorted(universo.keys())
    ano_ini, sem_ini = periodos[0]
    ano_fin, sem_fin = periodos[-1]
    primer_dia  = pd.Timestamp(ano_ini, 1 if sem_ini == 1 else 7, 1)
    ultimo_dia  = pd.Timestamp(ano_fin, 6 if sem_fin == 1 else 12, 30 if sem_fin == 1 else 31)
    inicios_mes = pd.date_range(start=primer_dia, end=ultimo_dia, freq="MS")

    tramos: list[pd.Series] = []
    valor_actual = 100.0

    print("\n" + "=" * 65)
    print("BACKTEST — ESTRATEGIA 12-1  (1 mes / Top 15%)")
    print("=" * 65)

    for mes_ini_cal in inicios_mes:
        ano = mes_ini_cal.year
        sem = 1 if mes_ini_cal.month <= 6 else 2

        # Fix #1: eliminar look-ahead bias — usar ultimo semestre publicado
        # S1 del anno Y (ene-jun): usar S2 del anno Y-1 (disponible ~feb-Y)
        # S2 del anno Y (jul-dic): usar S1 del anno Y   (disponible ~ago-Y)
        # S1 (ene-jun) publicado ~agosto → usarlo solo desde agosto; julio sigue con S2
        clave_universo = (ano - 1, 2) if mes_ini_cal.month <= 7 else (ano, 1)
        if clave_universo not in universo:
            continue

        fin_mes_cal = mes_ini_cal + pd.offsets.MonthEnd(0)
        dias_mes = precios.index[
            (precios.index >= mes_ini_cal) & (precios.index <= fin_mes_cal)
        ]
        if dias_mes.empty:
            continue

        fecha_ini = dias_mes[0]
        fecha_fin = dias_mes[-1]

        tickers_periodo = [t.upper() for t in universo[clave_universo]]
        momentos = calcular_momento_12_1(precios, tickers_periodo, fecha_ini)
        cartera  = seleccionar_top(momentos, TOP_PCT)
        if not cartera:
            continue

        # Fix #5: pct_change sobre ventana minima — 1 fila de ancla + mes actual
        idx_ini = precios.index.searchsorted(fecha_ini)
        idx_fin = precios.index.searchsorted(fecha_fin, side="right")
        ancla   = max(0, idx_ini - 1)
        ventana = precios.iloc[ancla:idx_fin][cartera]

        # Fix #4: fillna(0.0) blinda contra NaN si todos los tickers fallan un dia
        ret_diarios = ventana.pct_change().loc[fecha_ini:fecha_fin]
        ret_cartera = ret_diarios.mean(axis=1).fillna(0.0)

        equity_tramo = valor_actual * (1 + ret_cartera).cumprod()
        tramos.append(equity_tramo)
        valor_actual = float(equity_tramo.iloc[-1])

        ret_mes = equity_tramo.iloc[-1] / equity_tramo.iloc[0] - 1
        print(f"  [{ano}-{mes_ini_cal.month:02d}]  {fecha_ini.date()} -> {fecha_fin.date()}  "
              f"n={len(cartera)}  ret={ret_mes:+.2%}  equity={valor_actual:.1f}")

    if not tramos:
        return pd.Series(dtype=float)
    equity = pd.concat(tramos)
    return equity[~equity.index.duplicated(keep="last")]


# ===============================================================================
# 4. RESULTADOS Y GRAFICO
# ===============================================================================

def _recortar_normalizar(serie: pd.Series, fecha_ini: pd.Timestamp, fecha_fin: pd.Timestamp) -> pd.Series:
    r = serie[(serie.index >= fecha_ini) & (serie.index <= fecha_fin)]
    return (r / r.iloc[0] * 100) if not r.empty else r


def _calcular_metricas_dict(equity: pd.Series) -> dict:
    """Devuelve metricas como diccionario (para uso interno del dashboard)."""
    if equity.empty:
        return {"ret": np.nan, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    equity = equity.astype(float).dropna()
    v0, v1 = float(equity.iloc[0]), float(equity.iloc[-1])
    n_anos = (equity.index[-1] - equity.index[0]).days / 365.25
    ret_d  = equity.pct_change().dropna()
    vol    = float(ret_d.std()) * np.sqrt(252)
    cagr   = (v1 / v0) ** (1 / n_anos) - 1
    return {
        "ret":    v1 / v0 - 1,
        "cagr":   cagr,
        "vol":    vol,
        "sharpe": (float(ret_d.mean()) * 252) / vol if vol > 0 else np.nan,
        "maxdd":  float(((equity - equity.cummax()) / equity.cummax()).min()),
    }


def _calcular_alpha_beta(equity: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Calcula Alpha anualizado y Beta de la estrategia vs benchmark (ambas base 100)."""
    if equity.empty or benchmark.empty:
        return np.nan, np.nan
    idx = equity.index.intersection(benchmark.index)
    if len(idx) < 20:
        return np.nan, np.nan
    re = equity.reindex(idx).pct_change().dropna()
    rb = benchmark.reindex(idx).pct_change().dropna()
    idx2 = re.index.intersection(rb.index)
    re, rb = re.reindex(idx2).values, rb.reindex(idx2).values
    var_b = float(np.var(rb))
    if var_b == 0:
        return np.nan, np.nan
    beta  = float(np.cov(re, rb)[0, 1] / var_b)
    alpha = (float(np.mean(re)) - beta * float(np.mean(rb))) * 252
    return alpha, beta


def _metricas(equity: pd.Series, nombre: str) -> None:
    if equity.empty:
        return
    if isinstance(equity, pd.DataFrame):
        equity = equity.squeeze()
    equity = equity.astype(float).dropna()
    v0, v1  = float(equity.iloc[0]), float(equity.iloc[-1])
    n_anos  = (equity.index[-1] - equity.index[0]).days / 365.25
    ret_d   = equity.pct_change().dropna()
    vol     = float(ret_d.std()) * np.sqrt(252)
    cagr    = (v1 / v0) ** (1 / n_anos) - 1
    sharpe  = (float(ret_d.mean()) * 252) / vol if vol > 0 else np.nan
    max_dd  = float(((equity - equity.cummax()) / equity.cummax()).min())

    print(f"\n{nombre}")
    print(f"  Periodo:           {equity.index[0].date()} -> {equity.index[-1].date()}")
    print(f"  Retorno total:     {v1/v0-1:+.2%}")
    print(f"  CAGR:              {cagr:+.2%}")
    print(f"  Volatilidad anual: {vol:.2%}")
    print(f"  Sharpe (anual):    {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd:.2%}")


def mostrar_resultados(equity_est: pd.Series, nav_fondo: pd.Series, benchmark: pd.Series) -> None:
    """Imprime metricas de las 3 series y genera el grafico comparativo."""
    fecha_ini = equity_est.index[0]
    fecha_fin = equity_est.index[-1]

    cobas_plot = _recortar_normalizar(nav_fondo,  fecha_ini, fecha_fin)
    bm_plot    = _recortar_normalizar(benchmark,  fecha_ini, fecha_fin)

    print("\n" + "=" * 65)
    print("RESULTADOS FINALES")
    print("=" * 65)
    _metricas(equity_est, "Estrategia 12-1  (1 mes / Top 15%)")
    _metricas(cobas_plot, "COBAS Seleccion FI")
    _metricas(bm_plot,    "MSCI EAFE Small-Cap (SCZ)")

    # ── Grafico ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 1, height_ratios=[6, 0.5],
                           hspace=0.12, left=0.07, right=0.97, top=0.93, bottom=0.05)
    ax     = fig.add_subplot(gs[0])
    ax_per = fig.add_subplot(gs[1])

    if not equity_est.empty:
        ax.plot(equity_est.index, equity_est.values,
                label="Strategy 12-1 (1M / Top 15%)",
                color=COLOR_ESTRATEGIA, linewidth=2)

    if not cobas_plot.empty:
        ax.plot(cobas_plot.index, cobas_plot.values,
                label="COBAS Seleccion FI",
                color=COLOR_COBAS, linewidth=2)

    if not bm_plot.empty:
        ax.plot(bm_plot.index, bm_plot.values,
                label="MSCI EAFE Small-Cap (SCZ)",
                color=COLOR_BM, linewidth=2)

    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Strategy 12-1 (1M / Top 15%) vs. COBAS Seleccion FI vs. MSCI EAFE Small-Cap",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Value (base 100)", fontsize=11)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ── Period returns bar ────────────────────────────────────────────────────
    def _period_ret(series: pd.Series, end: pd.Timestamp, months) -> str:
        s = series.dropna()
        mask_e = s.index <= end
        if not mask_e.any():
            return "—"
        v_end = float(s[mask_e].iloc[-1])
        if months is None:
            v_start = float(s.iloc[0])
        else:
            start  = end - pd.DateOffset(months=months)
            mask_s = s.index <= start
            if not mask_s.any():
                return "—"
            v_start = float(s[mask_s].iloc[-1])
        if v_start == 0 or np.isnan(v_start) or np.isnan(v_end):
            return "—"
        return f"{v_end / v_start - 1:+.1%}"

    periods_bar = [
        ("1Y",  12),
        ("3Y",  36),
        ("5Y",  60),
        ("10Y", 120),
        ("MAX", None),
    ]
    parts = [f"{label}: {_period_ret(equity_est, fecha_fin, m)}"
             for label, m in periods_bar]

    ax_per.axis("off")
    ax_per.text(0.5, 0.5, "   |   ".join(parts),
                transform=ax_per.transAxes,
                ha="center", va="center", fontsize=11, color="black",
                bbox=dict(facecolor="#f0f0f0", edgecolor="#cccccc",
                          boxstyle="round,pad=0.5", linewidth=1.2))

    ruta_grafico = OUTPUTS_DIR / "backtest_equity_curves.png"
    plt.savefig(ruta_grafico, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafico guardado en: {ruta_grafico}")


# ===============================================================================
# 5. DASHBOARD COMPLETO (4 paneles + barra de metricas)
# ===============================================================================

def mostrar_dashboard(equity_est: pd.Series, nav_fondo: pd.Series, benchmark: pd.Series) -> None:
    """
    Dashboard de 4 paneles con tema oscuro:
      Panel 1: Equity (base 1) + sombreado outperf/underperf vs SCZ
      Panel 2: Drawdown de la estrategia
      Panel 3: Barras de retorno mensual (verde/rojo)
      Panel 4: Heatmap de retornos mensuales (año × mes)
      Pie:     Barra de metricas (CAGR, Sharpe, MaxDD, Vol, Calmar, Alpha, Beta)
    """
    fecha_ini = equity_est.index[0]
    fecha_fin = equity_est.index[-1]

    cobas_plot = _recortar_normalizar(nav_fondo, fecha_ini, fecha_fin)
    bm_plot    = _recortar_normalizar(benchmark, fecha_ini, fecha_fin)

    # Base 1 para el panel de equity
    est_1   = equity_est / equity_est.iloc[0]
    bm_1    = bm_plot    / bm_plot.iloc[0]    if not bm_plot.empty    else pd.Series(dtype=float)
    cobas_1 = cobas_plot / cobas_plot.iloc[0] if not cobas_plot.empty else pd.Series(dtype=float)

    # Metricas
    m              = _calcular_metricas_dict(equity_est)
    alpha, beta    = _calcular_alpha_beta(equity_est, bm_plot)
    calmar         = abs(m["cagr"] / m["maxdd"]) if m["maxdd"] != 0 else np.nan

    plt.style.use("default")
    fig = plt.figure(figsize=(16, 14))
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[4, 2, 3, 0.5],
        hspace=0.50, wspace=0.28,
        left=0.07, right=0.97, top=0.93, bottom=0.05,
    )

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_dd  = fig.add_subplot(gs[1, :])
    ax_bar = fig.add_subplot(gs[2, 0])
    ax_hot = fig.add_subplot(gs[2, 1])
    ax_st  = fig.add_subplot(gs[3, :])

    # ── Panel 1: Equity + sombreado outperf/underperf vs SCZ ─────────────────
    if not bm_1.empty:
        idx_c = est_1.index.intersection(bm_1.index)
        e_c   = est_1.reindex(idx_c)
        s_c   = bm_1.reindex(idx_c)
        ax_eq.fill_between(idx_c, e_c, s_c, where=e_c >= s_c,
                           color="#27ae60", alpha=0.20, label="Outperformance")
        ax_eq.fill_between(idx_c, e_c, s_c, where=e_c < s_c,
                           color="#c0392b", alpha=0.20, label="Underperformance")
        ax_eq.plot(bm_1.index, bm_1.values, color=COLOR_BM,
                   linewidth=1.5, linestyle="--", label="MSCI EAFE Small-Cap (SCZ)")

    if not cobas_1.empty:
        ax_eq.plot(cobas_1.index, cobas_1.values, color=COLOR_COBAS,
                   linewidth=1.5, linestyle=":", label="COBAS Seleccion FI")

    ax_eq.plot(est_1.index, est_1.values, color=COLOR_ESTRATEGIA,
               linewidth=2, label="Estrategia 12-1 (1M / Top 15%)")
    ax_eq.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax_eq.set_title("Estrategia 12-1  (1M / Top 15%)  —  COBAS Seleccion FI  vs  MSCI EAFE Small-Cap",
                    fontsize=13, fontweight="bold")
    ax_eq.set_ylabel("Crecimiento de $1 invertido", fontsize=11)
    ax_eq.legend(fontsize=10)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_eq.xaxis.set_major_locator(mdates.YearLocator())
    ax_eq.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    dd = (equity_est - equity_est.cummax()) / equity_est.cummax() * 100
    ax_dd.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.40)
    ax_dd.plot(dd.index, dd.values, color="#c0392b", linewidth=0.9)
    ax_dd.axhline(0, color="gray", linewidth=0.8)
    ax_dd.set_title("Drawdown — Estrategia 12-1", fontsize=11, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.grid(True, alpha=0.3)

    # ── Panel 3: Retornos mensuales (barras) ──────────────────────────────────
    ret_m = equity_est.resample("ME").last().pct_change().dropna() * 100
    colores_bar = ["#27ae60" if v >= 0 else "#c0392b" for v in ret_m.values]
    ax_bar.bar(ret_m.index, ret_m.values, color=colores_bar, width=20, alpha=0.80)
    ax_bar.axhline(0, color="gray", linewidth=0.8)
    ax_bar.set_title("Retornos mensuales (%)", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Retorno (%)", fontsize=9)
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bar.xaxis.set_major_locator(mdates.YearLocator())
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Heatmap mensual (año × mes) ──────────────────────────────────
    df_h           = ret_m.to_frame("ret")
    df_h["year"]   = ret_m.index.year
    df_h["month"]  = ret_m.index.month
    pivot          = df_h.pivot(index="year", columns="month", values="ret")
    pivot.columns  = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    vals_ok = pivot.values[~np.isnan(pivot.values.astype(float))]
    vabs    = max(float(np.abs(vals_ok).max()), 1.0) if len(vals_ok) else 10.0

    im = ax_hot.imshow(pivot.values.astype(float), cmap="RdYlGn",
                       aspect="auto", vmin=-vabs, vmax=vabs)
    ax_hot.set_xticks(range(12))
    ax_hot.set_xticklabels(pivot.columns, fontsize=8)
    ax_hot.set_yticks(range(len(pivot.index)))
    ax_hot.set_yticklabels(pivot.index.astype(str), fontsize=8)
    ax_hot.set_title("Heatmap retornos mensuales (%)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax_hot, shrink=0.85)

    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(float(val)):
                tcolor = "white" if abs(float(val)) > vabs * 0.55 else "black"
                ax_hot.text(j, i, f"{float(val):.1f}%", ha="center", va="center",
                            fontsize=6.5, color=tcolor, fontweight="bold")

    # ── Barra de metricas ─────────────────────────────────────────────────────
    ax_st.axis("off")
    stats = (
        f"CAGR: {m['cagr']:+.2%}   |   "
        f"Sharpe: {m['sharpe']:.3f}   |   "
        f"Max DD: {m['maxdd']:.2%}   |   "
        f"Volatilidad: {m['vol']:.2%}   |   "
        f"Calmar: {calmar:.3f}   |   "
        f"Alpha vs SCZ: {alpha:+.2%}   |   "
        f"Beta vs SCZ: {beta:.3f}   |   "
        f"Periodo: {fecha_ini.strftime('%Y-%m-%d')}  →  {fecha_fin.strftime('%Y-%m-%d')}"
    )
    ax_st.text(0.5, 0.5, stats, transform=ax_st.transAxes,
               ha="center", va="center", fontsize=9.5, color="black",
               bbox=dict(facecolor="#f0f0f0", edgecolor="#cccccc",
                         boxstyle="round,pad=0.5", linewidth=1.2))

    for ax in [ax_eq, ax_dd, ax_bar]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ruta = OUTPUTS_DIR / "backtest_dashboard.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nDashboard guardado en: {ruta}")


# ===============================================================================
# MAIN
# ===============================================================================

if __name__ == "__main__":
    print("── Cargando universo de tickers ──")
    universo = cargar_universo(RUTA_TICKERS)
    print(f"  {len(universo)} periodos: {sorted(universo.keys())}")

    print("\n── Cargando NAV del fondo ──")
    nav_fondo = cargar_nav_fondo(RUTA_EXCEL)

    print("\n── Descargando precios de acciones ──")
    precios = descargar_precios(universo)

    equity_est = ejecutar_backtest_mensual(universo, precios)

    print("\n── Descargando MSCI EAFE Small-Cap (SCZ) ──")
    benchmark = descargar_benchmark(equity_est.index[0], equity_est.index[-1])

    mostrar_resultados(equity_est, nav_fondo, benchmark)
    mostrar_dashboard(equity_est, nav_fondo, benchmark)
