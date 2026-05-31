"""
Borda Count mensual — Estrategia 12-1
Para cada mes natural, rankea las 100 combinaciones (Freq x Top N%) por
rentabilidad mensual. La mejor se lleva N puntos, la 2ª N-1, etc.
Al final acumula los puntos de todos los meses y muestra qué combinación
gana de forma más consistente, independientemente de su retorno total.
"""

import importlib.util
import os
import pickle
import sys, io
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
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
]
TOPS        = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TOP_LABELS  = [f"{int(p*100)}%" for p in TOPS]
FREQ_LABELS = [f[0] for f in FRECUENCIAS]
N_COMBOS    = len(FRECUENCIAS) * len(TOPS)

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

    equities = {}
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

    # ── Retornos mensuales de cada combinacion ─────────────────────────────────
    ret_mensuales = {}
    for combo, eq in equities.items():
        if eq.empty:
            continue
        mensual = eq.resample("ME").last().pct_change().dropna()
        ret_mensuales[combo] = mensual

    df_ret = pd.DataFrame(ret_mensuales)
    df_ret = df_ret.dropna(how="any")

    n_meses  = len(df_ret)
    n_combos = len(df_ret.columns)
    print(f"Meses con datos: {n_meses}  |  Combinaciones: {n_combos}\n")

    # ── Borda Count: rankear por mes y acumular puntos ─────────────────────────
    puntos_df      = df_ret.rank(axis=1, ascending=False, method="min")
    puntos_df      = n_combos - puntos_df + 1
    puntos_totales = puntos_df.sum()
    puntos_max     = n_combos * n_meses

    # ── R² de linealidad: mide estacionariedad del ranking ────────────────────
    # R² alto (≈1) significa que la estrategia acumula puntos a ritmo constante
    # a lo largo de todo el período → su ranking relativo no varía con el tiempo.
    x_norm = np.arange(n_meses, dtype=float)
    r2_scores = {}
    for combo in puntos_df.columns:
        acum = puntos_df[combo].cumsum().values.astype(float)
        r2_scores[combo] = float(np.corrcoef(x_norm, acum)[0, 1] ** 2)

    # ── Tabla top 20 por puntos totales ───────────────────────────────────────
    ranking = (
        puntos_totales
        .reset_index()
        .rename(columns={"level_0": "freq", "level_1": "top", 0: "puntos"})
        .sort_values("puntos", ascending=False)
    )

    print("=" * 70)
    print(f"TOP 20 COMBINACIONES POR BORDA COUNT  ({n_meses} meses)")
    print("=" * 70)
    print(f"  {'Pos':>3}  {'Freq':<6} {'Top%':<6} {'Puntos':>8} {'% del max':>10}  {'σ rank':>8}  {'R²':>6}")
    print("  " + "-" * 65)
    for pos, (_, row) in enumerate(ranking.head(20).iterrows(), start=1):
        combo    = (row["freq"], row["top"])
        pct_m    = row["puntos"] / puntos_max * 100
        rank_std = float(puntos_df[combo].std())
        r2       = r2_scores.get(combo, float("nan"))
        print(f"  {pos:>3}  {row['freq']:<6} {row['top']:<6} {row['puntos']:>8.0f} "
              f"{pct_m:>9.1f}%  {rank_std:>8.1f}  {r2:>6.4f}")
    print("=" * 70)

    # ── Heatmap de puntos totales ──────────────────────────────────────────────
    grid_puntos  = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)
    grid_rankstd = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)

    for fl, tl in equities.keys():
        combo = (fl, tl)
        if combo in puntos_totales.index:
            grid_puntos.loc[fl, tl]  = float(puntos_totales[combo])
            grid_rankstd.loc[fl, tl] = float(puntos_df[combo].std())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, titulo, fmt, cmap in [
        (axes[0], grid_puntos,  f"Puntos Borda acumulados ({n_meses} meses)", "%.0f",  "RdYlGn"),
        (axes[1], grid_rankstd, "σ ranking mensual (menor = más consistente)", "%.1f", "RdYlGn_r"),
    ]:
        vals = data.values.astype(float)
        im   = ax.imshow(vals, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(TOP_LABELS)))
        ax.set_xticklabels(TOP_LABELS, fontsize=10)
        ax.set_yticks(range(len(FREQ_LABELS)))
        ax.set_yticklabels(FREQ_LABELS, fontsize=10)
        ax.set_xlabel("Top N%", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        for i in range(len(FREQ_LABELS)):
            for j in range(len(TOP_LABELS)):
                val = float(data.iloc[i, j])
                if not np.isnan(val):
                    norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                    tc   = "white" if norm < 0.20 or norm > 0.80 else "black"
                    ax.text(j, i, fmt % val, ha="center", va="center",
                            fontsize=8.5, color=tc, fontweight="bold")

    plt.suptitle(
        f"Borda Count Mensual — Frecuencia x Top N%  ({n_meses} meses evaluados)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    ruta1 = OUTPUTS / "borda_heatmap.png"
    plt.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfico 1 guardado: {ruta1}")

    # ── Puntos acumulados en el tiempo — top 5 combinaciones ──────────────────
    top5    = [(row["freq"], row["top"]) for _, row in ranking.head(5).iterrows()]
    colores = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(15, 6))
    for (fl, tl), color in zip(top5, colores):
        combo = (fl, tl)
        if combo not in puntos_df.columns:
            continue
        acum  = puntos_df[combo].cumsum()
        total = float(puntos_totales[combo])
        ax.plot(acum.index, acum.values,
                label=f"{fl} / {tl}  ({total:.0f} pts)", color=color, linewidth=1.8)

    ax.set_title("Puntos Borda acumulados en el tiempo — Top 5 combinaciones",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Puntos acumulados", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    ruta2 = OUTPUTS / "borda_acumulado.png"
    plt.savefig(ruta2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfico 2 guardado: {ruta2}")
