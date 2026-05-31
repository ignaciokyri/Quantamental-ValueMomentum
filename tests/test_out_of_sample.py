"""
Validación Out-of-Sample — Walk-Forward Expanding Window

Metodología
───────────
Para evitar el sesgo de data snooping al elegir 1M/15% *mirando todos los datos*,
este test reconstruye cómo habría ido la estrategia si hubiéramos re-elegido
los parámetros periódicamente usando solo el pasado disponible en cada momento.

  1. Ventana de entrenamiento mínima: MIN_TRAIN_MONTHS (primer corte).
  2. Cada OOS_STEP_MONTHS: se selecciona la combinación (Freq × Top N%) que
     habría tenido mayor Sharpe anualizado en TODO el historial disponible hasta
     ese punto — sin tocar un solo dato futuro.
  3. Esa combinación se aplica en los próximos OOS_STEP_MONTHS.
  4. Los tramos OOS se encadenan en una sola curva de equidad "adaptativa".

Comparaciones
─────────────
  - Walk-forward OOS   → el resultado real de este procedimiento adaptativo
  - Fijo 1M/15%        → el óptimo in-sample aplicado de forma fija
  - MSCI EAFE SC (SCZ) → benchmark de mercado
  - COBAS Selección FI → fondo original

Si la curva OOS se acerca a la fija in-sample, los parámetros son estables
y el resultado no es un artefacto del ajuste de curvas.
Si colapsa, el buen resultado histórico era sobreajuste.

Outputs
───────
  outputs/oos_equity.png — curva OOS vs comparaciones
  Tabla consola con los parámetros elegidos en cada ventana
"""

import importlib.util
import os
import pickle
import sys, io
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"

FRECUENCIAS = [
    ("1S",   "W-MON"),
    ("2S",   "2W-MON"),
    ("3S",   "3W-MON"),
    ("4S",   "4W-MON"),
    ("5S",   "5W-MON"),
    ("6S",   "6W-MON"),
    ("7S",   "7W-MON"),
    ("8S",   "8W-MON"),
    ("9S",   "9W-MON"),
    ("10S",  "10W-MON"),
    ("1M",   "MS"),
]
TOPS       = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TOP_LABELS = [f"{int(p*100)}%" for p in TOPS]
N_COMBOS   = len(FRECUENCIAS) * len(TOPS)

MIN_TRAIN_MONTHS = 36   # al menos 3 años de historial antes del primer test
OOS_STEP_MONTHS  = 12   # re-seleccionar parámetros una vez al año


# ── Helpers (module-level para ser picklables) ─────────────────────────────────

def _cargar_bt():
    spec = importlib.util.spec_from_file_location("bt", BASE / "main" / "backtest.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.stdout = io.StringIO()
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = sys.__stdout__
    return mod


_proc = {}

def _init_worker(universo_pkl, precios_pkl):
    _proc["universo"] = pickle.loads(universo_pkl)
    _proc["precios"]  = pickle.loads(precios_pkl)
    _proc["bt"]       = _cargar_bt()


def _run_combo(args):
    fl, fs, pct, tl = args
    eq = _proc["bt"].backtest_parametrico(_proc["universo"], _proc["precios"], fs, pct)
    return fl, tl, eq


def _sharpe_mensual(eq_mensual: pd.Series) -> float:
    """Sharpe anualizado a partir de una serie de equity con frecuencia mensual."""
    r = eq_mensual.pct_change().dropna()
    if len(r) < 6 or float(r.std()) == 0:
        return -np.inf
    return float(r.mean() / r.std() * np.sqrt(12))


def _metricas(eq: pd.Series) -> dict:
    """CAGR, Sharpe diario anualizado y MaxDD de una curva de equidad diaria."""
    if eq.empty or len(eq) < 5:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    eq  = eq.astype(float).dropna()
    r   = eq.pct_change().dropna()
    v0, v1 = float(eq.iloc[0]), float(eq.iloc[-1])
    n   = (eq.index[-1] - eq.index[0]).days / 365.25
    vol = float(r.std()) * np.sqrt(252)
    return {
        "cagr":   float((v1 / v0) ** (1 / n) - 1) if n > 0 else np.nan,
        "sharpe": (float(r.mean()) * 252) / vol if vol > 0 else np.nan,
        "maxdd":  float(((eq - eq.cummax()) / eq.cummax()).min()),
    }


def _norm(s: pd.Series, ini: pd.Timestamp, fin: pd.Timestamp, base: float = 100.0) -> pd.Series:
    s = s[(s.index >= ini) & (s.index <= fin)].dropna()
    return (s / s.iloc[0] * base) if not s.empty else s


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    bt = _cargar_bt()
    print("Cargando universo y precios...")
    universo = bt.cargar_universo(bt.RUTA_TICKERS)
    precios  = bt.descargar_precios(universo)
    print("Listo.\n")

    u_pkl = pickle.dumps(universo)
    p_pkl = pickle.dumps(precios)

    tasks     = [(fl, fs, pct, tl) for fl, fs in FRECUENCIAS
                                    for pct, tl in zip(TOPS, TOP_LABELS)]
    n_workers = min(os.cpu_count() or 4, N_COMBOS)

    equities: dict[tuple, pd.Series] = {}
    done = 0

    print(f"Ejecutando {N_COMBOS} backtests en paralelo ({n_workers} workers)...\n")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(u_pkl, p_pkl),
    ) as pool:
        futures = {pool.submit(_run_combo, t): t for t in tasks}
        for future in as_completed(futures):
            fl, tl, eq = future.result()
            equities[(fl, tl)] = eq
            done += 1
            print(f"  [{done:>3}/{N_COMBOS}] {fl} x {tl}", end="\r")

    print(f"\nBacktests completados.\n")

    # ── Resample mensual para el criterio de selección ────────────────────────
    eq_mens: dict[tuple, pd.Series] = {}
    for combo, eq in equities.items():
        if not eq.empty:
            eq_mens[combo] = eq.resample("ME").last().dropna()

    fechas_globales = sorted({f for eq in eq_mens.values() for f in eq.index})
    fecha_ini_global = fechas_globales[0]
    fecha_fin_global = fechas_globales[-1]

    # ── Walk-forward: cortes al inicio de cada ventana OOS ───────────────────
    cortes_oos = pd.date_range(
        start=fecha_ini_global + pd.DateOffset(months=MIN_TRAIN_MONTHS),
        end=fecha_fin_global,
        freq=f"{OOS_STEP_MONTHS}MS",
    )

    print("=" * 78)
    print(f"WALK-FORWARD  (entrenamiento mínimo = {MIN_TRAIN_MONTHS}M, ventana OOS = {OOS_STEP_MONTHS}M)")
    print("=" * 78)
    print(f"  {'#':>3}  {'Train hasta':>11}  {'OOS':>22}  {'Combo':>10}  "
          f"{'IS Sharpe':>10}  {'OOS ret':>8}")
    print("  " + "-" * 71)

    tramos_oos  : list[pd.Series] = []
    selecciones : list[tuple]     = []

    for i, corte in enumerate(cortes_oos):
        oos_ini = corte
        oos_fin = min(corte + pd.DateOffset(months=OOS_STEP_MONTHS) - pd.Timedelta(days=1),
                      fecha_fin_global)

        # Selección: mejor Sharpe sobre historial completo ANTES del corte
        mejor_combo  = None
        mejor_sharpe = -np.inf
        for combo, eq_m in eq_mens.items():
            train = eq_m[eq_m.index < oos_ini]
            if len(train) < MIN_TRAIN_MONTHS // 3:
                continue
            sh = _sharpe_mensual(train)
            if sh > mejor_sharpe:
                mejor_sharpe = sh
                mejor_combo  = combo

        if mejor_combo is None:
            continue

        selecciones.append(mejor_combo)

        # Tramo OOS: curva diaria del combo ganador en el período test
        eq_diaria = equities[mejor_combo]
        oos_slice = eq_diaria[(eq_diaria.index >= oos_ini) & (eq_diaria.index <= oos_fin)]
        if oos_slice.empty:
            continue

        # Encadenar: normalizar al último valor del tramo anterior
        escala = float(tramos_oos[-1].iloc[-1]) if tramos_oos else 100.0
        oos_norm = oos_slice / float(oos_slice.iloc[0]) * escala
        tramos_oos.append(oos_norm)

        oos_ret     = float(oos_slice.iloc[-1] / oos_slice.iloc[0] - 1)
        train_label = (oos_ini - pd.Timedelta(days=1)).strftime("%Y-%m")
        oos_label   = f"{oos_ini.strftime('%Y-%m')} - {oos_fin.strftime('%Y-%m')}"
        fl, tl = mejor_combo
        print(f"  {i+1:>3}  {train_label:>11}  {oos_label:>22}  "
              f"{fl+'/'+tl:>10}  {mejor_sharpe:>+10.3f}  {oos_ret:>+8.1%}")

    print("=" * 78)

    if not tramos_oos:
        print("Sin datos suficientes para el walk-forward.")
        raise SystemExit

    eq_oos = pd.concat(tramos_oos)
    eq_oos = eq_oos[~eq_oos.index.duplicated(keep="last")]

    # ── Parámetros más seleccionados ──────────────────────────────────────────
    conteo      = Counter(selecciones)
    n_ventanas  = len(selecciones)
    print(f"\nParámetros seleccionados en {n_ventanas} ventanas OOS:")
    for combo, n in conteo.most_common(10):
        print(f"  {combo[0]}/{combo[1]:>4}  → {n}/{n_ventanas} veces ({n/n_ventanas:.0%})")

    # Concentración: si el top-1 aparece >50% es señal de estabilidad
    top1_pct = conteo.most_common(1)[0][1] / n_ventanas
    print(f"\n  Concentración top-1: {top1_pct:.0%}  "
          f"({'parámetros estables' if top1_pct >= 0.50 else 'parámetros rotativos'})")

    # ── Métricas comparadas ───────────────────────────────────────────────────
    fecha_oos_ini = eq_oos.index[0]
    fecha_oos_fin = eq_oos.index[-1]

    eq_fixed = _norm(equities.get(("1M", "15%"), pd.Series(dtype=float)),
                     fecha_oos_ini, fecha_oos_fin)
    nav_fondo = bt.cargar_nav_fondo(bt.RUTA_EXCEL)
    cobas_oos = _norm(nav_fondo, fecha_oos_ini, fecha_oos_fin)

    print("\n── Descargando benchmark SCZ ──")
    bm_raw = bt.descargar_benchmark(fecha_oos_ini, fecha_oos_fin)
    bm_oos = _norm(bm_raw, fecha_oos_ini, fecha_oos_fin)

    m_oos   = _metricas(eq_oos)
    m_fixed = _metricas(eq_fixed)
    m_bm    = _metricas(bm_oos)
    m_cobas = _metricas(cobas_oos)

    print(f"\n{'='*65}")
    print(f"METRICAS OOS  ({fecha_oos_ini.date()} → {fecha_oos_fin.date()})")
    print(f"{'='*65}")
    print(f"  {'Estrategia':28}  {'CAGR':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
    print(f"  {'-'*56}")
    for nombre, m in [
        ("Walk-forward adaptativo",    m_oos),
        ("Fijo 1M/15% (IS-óptimo)",    m_fixed),
        ("MSCI EAFE Small-Cap (SCZ)",   m_bm),
        ("COBAS Selección FI",          m_cobas),
    ]:
        cagr_s   = f"{m['cagr']:>+8.2%}"    if not np.isnan(m['cagr'])   else "     —"
        sharpe_s = f"{m['sharpe']:>8.3f}"   if not np.isnan(m['sharpe']) else "     —"
        maxdd_s  = f"{m['maxdd']:>8.2%}"    if not np.isnan(m['maxdd'])  else "     —"
        print(f"  {nombre:28}  {cagr_s}  {sharpe_s}  {maxdd_s}")
    print(f"{'='*65}")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sombrear las ventanas OOS alternas
    for j, corte in enumerate(cortes_oos):
        ini_s = max(corte, fecha_oos_ini)
        fin_s = min(corte + pd.DateOffset(months=OOS_STEP_MONTHS), fecha_oos_fin)
        if ini_s >= fin_s:
            continue
        if j % 2 == 0:
            ax.axvspan(ini_s, fin_s, color="#f0f0f0", alpha=0.6, zorder=0)

    if not bm_oos.empty:
        ax.plot(bm_oos.index, bm_oos.values, color="#2980b9",
                linewidth=1.4, linestyle=":", label="MSCI EAFE Small-Cap (SCZ)")

    if not cobas_oos.empty:
        ax.plot(cobas_oos.index, cobas_oos.values, color="#27ae60",
                linewidth=1.4, linestyle="-.", label="COBAS Selección FI")

    if not eq_fixed.empty:
        ax.plot(eq_fixed.index, eq_fixed.values, color="#c0392b",
                linewidth=1.8, linestyle="--", label="Fijo 1M/15% (óptimo in-sample)")

    ax.plot(eq_oos.index, eq_oos.values, color="#8e44ad",
            linewidth=2.5, label="Walk-forward OOS (parámetros adaptados)")

    # Marcar los cortes de re-entrenamiento
    for corte in cortes_oos:
        if fecha_oos_ini <= corte <= fecha_oos_fin:
            ax.axvline(corte, color="#555555", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(
        f"Validación Out-of-Sample — Walk-forward "
        f"({MIN_TRAIN_MONTHS}M train, {OOS_STEP_MONTHS}M OOS por ventana)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Valor (base 100)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    ruta = OUTPUTS / "oos_equity.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfico guardado: {ruta}")
