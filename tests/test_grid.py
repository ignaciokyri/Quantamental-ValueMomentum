"""
Grid search: frecuencia de rebalanceo x Top N%
Prueba todas las combinaciones y genera heatmaps + curvas de equity.
"""

import importlib.util
import os
import pickle
import sys, io
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"

FRECUENCIAS = [
    ("1S",  "W-MON"),
    ("2S",  "2W-MON"),
    ("3S",  "3W-MON"),
    ("4S",  "4W-MON"),
    ("5S",  "5W-MON"),
    ("6S",  "6W-MON"),
    ("7S",  "7W-MON"),
    ("8S",  "8W-MON"),
    ("9S",  "9W-MON"),
    ("10S", "10W-MON"),
]
TOPS        = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TOP_LABELS  = [f"{int(p*100)}%" for p in TOPS]
FREQ_LABELS = [f[0] for f in FRECUENCIAS]

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


def metricas(eq: pd.Series) -> dict:
    if eq.empty:
        return {k: np.nan for k in ["cagr", "sharpe", "maxdd", "ret", "vol"]}
    eq  = eq.astype(float).dropna()
    v0, v1 = float(eq.iloc[0]), float(eq.iloc[-1])
    n   = (eq.index[-1] - eq.index[0]).days / 365.25
    rd  = eq.pct_change().dropna()
    vol = float(rd.std()) * np.sqrt(252)
    return {
        "ret":    v1 / v0 - 1,
        "cagr":   (v1 / v0) ** (1 / n) - 1 if n > 0 else np.nan,
        "vol":    vol,
        "sharpe": (float(rd.mean()) * 252) / vol if vol > 0 else np.nan,
        "maxdd":  float(((eq - eq.cummax()) / eq.cummax()).min()),
    }


# ── Worker: se inicializa una vez por proceso, carga datos y modulo ────────────

_proc = {}

def _init_worker(universo_pkl, precios_pkl):
    _proc["universo"] = pickle.loads(universo_pkl)
    _proc["precios"]  = pickle.loads(precios_pkl)
    _proc["bt"]       = _cargar_bt()


def _run_combo(args):
    fl, fs, pct, tl = args
    eq = _proc["bt"].backtest_parametrico(_proc["universo"], _proc["precios"], fs, pct)
    return fl, tl, eq, metricas(eq)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    bt = _cargar_bt()
    print("Cargando universo y precios...")
    universo = bt.cargar_universo(bt.RUTA_TICKERS)
    precios  = bt.descargar_precios(universo)
    print("Listo.\n")

    # Serializar una sola vez para el initializer
    u_pkl = pickle.dumps(universo)
    p_pkl = pickle.dumps(precios)

    tasks    = [(fl, fs, pct, tl) for fl, fs in FRECUENCIAS
                                   for pct, tl in zip(TOPS, TOP_LABELS)]
    n_combos = len(tasks)
    n_workers = min(os.cpu_count() or 4, n_combos)

    grid_cagr   = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)
    grid_sharpe = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)
    grid_maxdd  = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)
    grid_ret    = pd.DataFrame(index=FREQ_LABELS, columns=TOP_LABELS, dtype=float)
    equities    = {}

    print(f"Ejecutando {n_combos} combinaciones en paralelo ({n_workers} workers)...\n")

    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(u_pkl, p_pkl),
    ) as pool:
        futures = {pool.submit(_run_combo, t): t for t in tasks}
        for future in as_completed(futures):
            fl, tl, eq, m = future.result()
            grid_cagr.loc[fl, tl]   = m["cagr"]
            grid_sharpe.loc[fl, tl] = m["sharpe"]
            grid_maxdd.loc[fl, tl]  = m["maxdd"]
            grid_ret.loc[fl, tl]    = m["ret"]
            equities[(fl, tl)]      = eq
            done += 1
            print(f"  [{done:>2}/{n_combos}] {fl} x {tl}  "
                  f"CAGR={m['cagr']:+.2%}  Sharpe={m['sharpe']:.2f}  MaxDD={m['maxdd']:.2%}")

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    for titulo, grid_rank, key in [
        ("TOP 10 COMBINACIONES POR CAGR",   grid_cagr,   "cagr"),
        ("TOP 10 COMBINACIONES POR SHARPE", grid_sharpe, "sharpe"),
    ]:
        print("\n" + "=" * 65)
        print(titulo)
        print("=" * 65)
        print(f"{'Freq':<6} {'Top%':<6} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9} {'Ret.Total':>10}")
        print("-" * 65)
        ranking = (
            grid_rank.stack()
            .reset_index()
            .rename(columns={"level_0": "freq", "level_1": "top", 0: key})
            .sort_values(key, ascending=False)
            .head(10)
        )
        for _, row in ranking.iterrows():
            cg = grid_cagr.loc[row.freq, row.top]
            sh = grid_sharpe.loc[row.freq, row.top]
            dd = grid_maxdd.loc[row.freq, row.top]
            rt = grid_ret.loc[row.freq, row.top]
            print(f"{row.freq:<6} {row.top:<6} {cg:>8.2%} {sh:>8.2f} {dd:>9.2%} {rt:>10.2%}")
        print("=" * 65)

    # ══════════════════════════════════════════════════════════════════════════
    # GRAFICOS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Grafico 1: Heatmaps CAGR y Sharpe ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, titulo, fmt in [
        (axes[0], grid_cagr * 100, "CAGR (%)", "%.1f%%"),
        (axes[1], grid_sharpe,     "Sharpe",   "%.2f"),
    ]:
        im = ax.imshow(data.values.astype(float), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(TOP_LABELS)));  ax.set_xticklabels(TOP_LABELS, fontsize=10)
        ax.set_yticks(range(len(FREQ_LABELS))); ax.set_yticklabels(FREQ_LABELS, fontsize=10)
        ax.set_xlabel("Top N%", fontsize=11);   ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
        vmin, vmax = data.values.astype(float).min(), data.values.astype(float).max()
        for i in range(len(FREQ_LABELS)):
            for j in range(len(TOP_LABELS)):
                val  = float(data.iloc[i, j])
                norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                tc   = "white" if norm < 0.25 or norm > 0.80 else "black"
                ax.text(j, i, fmt % val, ha="center", va="center",
                        fontsize=8.5, color=tc, fontweight="bold")
    plt.suptitle("Grid Search — Frecuencia x Top N% (Estrategia 12-1)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    ruta1 = OUTPUTS / "grid_heatmaps.png"
    plt.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafico 1 guardado: {ruta1}")

    # ── Grafico 2: Top 5 + Bottom 5 curvas de equity ─────────────────────────
    all_ranked = (
        grid_cagr.stack().reset_index()
        .rename(columns={"level_0": "freq", "level_1": "top", 0: "cagr"})
        .sort_values("cagr", ascending=False)
    )
    top5_combos    = [(r.freq, r.top) for _, r in all_ranked.head(5).iterrows()]
    bottom5_combos = [(r.freq, r.top) for _, r in all_ranked.tail(5).iterrows()]
    ref            = ("4S", "20%")

    COLORES_TOP    = ["#1a5276", "#2980b9", "#5dade2", "#aed6f1", "#d6eaf8"]
    COLORES_BOTTOM = ["#7b241c", "#c0392b", "#e74c3c", "#f1948a", "#fadbd8"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    for ax, combos, colores, titulo in [
        (axes[0], top5_combos,    COLORES_TOP,    "TOP 5 combinaciones por CAGR"),
        (axes[1], bottom5_combos, COLORES_BOTTOM, "BOTTOM 5 combinaciones por CAGR"),
    ]:
        for (fl, tl), color in zip(combos, colores):
            eq = equities.get((fl, tl))
            if eq is None or eq.empty:
                continue
            label = f"{fl}/{tl}  (CAGR={float(grid_cagr.loc[fl,tl]):.1%}, Sharpe={float(grid_sharpe.loc[fl,tl]):.2f})"
            ax.plot(eq.index, eq.values, label=label, color=color, linewidth=1.8)
        eq_ref = equities.get(ref)
        if eq_ref is not None and not eq_ref.empty:
            label = f"REF {ref[0]}/{ref[1]}  (CAGR={float(grid_cagr.loc[ref[0],ref[1]]):.1%})"
            ax.plot(eq_ref.index, eq_ref.values, label=label,
                    color="#f39c12", linewidth=2.2, linestyle="--")
        ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_ylabel("Valor (base 100)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)
    fig.suptitle("Grid Search — Top 5 vs Bottom 5 por CAGR",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.autofmt_xdate()
    plt.tight_layout()
    ruta2 = OUTPUTS / "grid_equity_top5.png"
    plt.savefig(ruta2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafico 2 guardado: {ruta2}")

    # ── Grafico 3: Heatmap MaxDD ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data_dd = grid_maxdd.abs() * 100
    im = ax.imshow(data_dd.values.astype(float), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(TOP_LABELS)));  ax.set_xticklabels(TOP_LABELS, fontsize=10)
    ax.set_yticks(range(len(FREQ_LABELS))); ax.set_yticklabels(FREQ_LABELS, fontsize=10)
    ax.set_xlabel("Top N%", fontsize=11);   ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title("Max Drawdown (%) — menor es mejor", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)
    vmin_dd, vmax_dd = data_dd.values.astype(float).min(), data_dd.values.astype(float).max()
    for i in range(len(FREQ_LABELS)):
        for j in range(len(TOP_LABELS)):
            val  = float(data_dd.iloc[i, j])
            norm = (val - vmin_dd) / (vmax_dd - vmin_dd) if vmax_dd != vmin_dd else 0.5
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if norm > 0.75 else "black", fontweight="bold")
    plt.tight_layout()
    ruta3 = OUTPUTS / "grid_maxdd.png"
    plt.savefig(ruta3, dpi=150)
    plt.show()
    print(f"Grafico 3 guardado: {ruta3}")

    # ── Grafico 4: Todas las curvas de equity ─────────────────────────────────
    all_combos = sorted(
        [(fl, tl, float(grid_cagr.loc[fl, tl]))
         for fl, tl in equities
         if not equities[(fl, tl)].empty and not np.isnan(float(grid_cagr.loc[fl, tl]))],
        key=lambda x: x[2],
    )
    cagr_vals = np.array([c[2] for c in all_combos])
    cmap      = plt.cm.RdYlGn
    norm      = plt.Normalize(vmin=cagr_vals.min(), vmax=cagr_vals.max())

    fig, ax = plt.subplots(figsize=(15, 7))
    for fl, tl, cagr_val in all_combos:
        ax.plot(equities[(fl, tl)].index, equities[(fl, tl)].values,
                color=cmap(norm(cagr_val)), linewidth=0.8, alpha=0.6)
    for fl, tl, cagr_val in [all_combos[-1], all_combos[0]]:
        ax.plot(equities[(fl, tl)].index, equities[(fl, tl)].values,
                color=cmap(norm(cagr_val)), linewidth=2.5,
                label=f"{fl}/{tl}  CAGR={cagr_val:.1%}")
    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("CAGR", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Todas las combinaciones — curvas de equity (color = CAGR)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor (base 100)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    ruta4 = OUTPUTS / "grid_equity_all.png"
    plt.savefig(ruta4, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafico 4 guardado: {ruta4}")
