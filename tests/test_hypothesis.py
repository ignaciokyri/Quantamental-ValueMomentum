"""
Contraste de hipotesis — Estrategia 12-1
¿Es 1M/15% estadisticamente el mejor parametro o podria ser azar?

Tests aplicados
───────────────
1. Block Bootstrap Sharpe CI
   Intervalo de confianza del 95% para el Sharpe de 1M/15% y sus rivales.
   Bloque = 3 meses (preserva autocorrelacion tipica del momentum mensual).

2. Test pareado por pares (H0: E[d_t] = 0)
   d_t = ret(1M/15%)_t − ret(rival)_t  para cada mes t.
   p-valor: fraccion de muestras bootstrap donde mean(d*) <= 0
   (test unilateral: ¿supera 1M/15% al rival?).

3. Correccion Benjamini-Yekutieli (BHY)
   Controla la tasa de falsos descubrimientos (FDR) bajo dependencia
   arbitraria entre tests — adecuado para retornos correlacionados.

4. White's Reality Check (simplificado)
   p-valor global: P(max_k V_k* > V_target) bajo H0.
   Corrige el sesgo de haber elegido 1M/15% *despues* de ver los resultados.

Outputs
───────
  outputs/hypothesis_pvalues.png   — heatmap p-valores BHY
  Tabla consola: top 20 rivales con p-valor, correccion BHY, significacion
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
    ("1M",  "MS"),      # target explicito
]
TOPS        = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TOP_LABELS  = [f"{int(p*100)}%" for p in TOPS]
FREQ_LABELS = [f[0] for f in FRECUENCIAS]
N_COMBOS    = len(FRECUENCIAS) * len(TOPS)

TARGET     = ("1M", "15%")
BLOCK_SIZE = 3
N_BOOT     = 3000
ALPHA      = 0.05

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

    # ── Retornos mensuales ─────────────────────────────────────────────────────
    ret_m = {}
    for combo, eq in equities.items():
        if not eq.empty:
            r = eq.resample("ME").last().pct_change().dropna()
            ret_m[combo] = r

    df_ret = pd.DataFrame(ret_m).dropna(how="any")

    if TARGET not in df_ret.columns:
        raise ValueError(f"Target {TARGET} no encontrado. Comprueba FRECUENCIAS y TOPS.")

    T = len(df_ret)
    print(f"Meses con datos: {T}\n")

    # ── Block Bootstrap ────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    def _sharpe_anual(series: np.ndarray) -> float:
        mu, sd = series.mean(), series.std()
        return (mu / sd * np.sqrt(12)) if sd > 0 else np.nan

    def _cagr_anual(series: np.ndarray) -> float:
        n = len(series)
        return float(np.prod(1 + series) ** (12 / n) - 1) if n > 0 else np.nan

    def block_bootstrap_sharpe(ret_matrix: np.ndarray) -> np.ndarray:
        """Devuelve array (N_BOOT, n_cols) con Sharpes anualizados bootstrapeados."""
        T_local, K = ret_matrix.shape
        sharpes_boot = np.zeros((N_BOOT, K))
        n_blocks = int(np.ceil(T_local / BLOCK_SIZE))

        for b in range(N_BOOT):
            starts = rng.integers(0, T_local, size=n_blocks)
            idx = np.concatenate([
                np.arange(s, s + BLOCK_SIZE) % T_local for s in starts
            ])[:T_local]
            sample = ret_matrix[idx]
            sharpes_boot[b] = np.array([_sharpe_anual(sample[:, k]) for k in range(K)])

        return sharpes_boot

    print(f"Block bootstrap ({N_BOOT} muestras, bloque={BLOCK_SIZE} meses)...")
    mat     = df_ret.values.astype(float)
    cols    = list(df_ret.columns)
    sh_boot = block_bootstrap_sharpe(mat)
    sh_obs  = np.array([_sharpe_anual(mat[:, k]) for k in range(len(cols))])
    print("Bootstrap completado.\n")

    # ── CI del target ──────────────────────────────────────────────────────────
    idx_target     = cols.index(TARGET)
    sh_target_boot = sh_boot[:, idx_target]
    ci_lo, ci_hi   = np.nanpercentile(sh_target_boot, [2.5, 97.5])
    sh_target_obs  = sh_obs[idx_target]

    rng_cagr   = np.random.default_rng(43)
    ret_target = mat[:, idx_target]
    cagr_obs   = _cagr_anual(ret_target)
    n_blocks_t = int(np.ceil(T / BLOCK_SIZE))
    cagr_boot  = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        starts        = rng_cagr.integers(0, T, size=n_blocks_t)
        idx_b         = np.concatenate([np.arange(s, s + BLOCK_SIZE) % T for s in starts])[:T]
        cagr_boot[b]  = _cagr_anual(ret_target[idx_b])
    cagr_ci_lo, cagr_ci_hi = np.nanpercentile(cagr_boot, [2.5, 97.5])

    print(f"Target: {TARGET[0]} / {TARGET[1]}")
    print(f"  Sharpe observado : {sh_target_obs:.3f}")
    print(f"  IC 95% Sharpe    : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  CAGR observado   : {cagr_obs:.2%}")
    print(f"  IC 95% CAGR      : [{cagr_ci_lo:.2%}, {cagr_ci_hi:.2%}]\n")

    # ── Test pareado H0: Sharpe(target) = Sharpe(rival) ──────────────────────
    pvalores_raw = {}

    for k, combo in enumerate(cols):
        if combo == TARGET:
            continue
        d_obs_k  = sh_obs[idx_target] - sh_obs[k]
        d_boot_k = sh_boot[:, idx_target] - sh_boot[:, k]
        # Centrar bajo H₀ (diferencia verdadera = 0) y calcular p unilateral
        p = float(np.mean(d_boot_k - d_boot_k.mean() >= d_obs_k))
        pvalores_raw[combo] = p

    # ── Correccion BHY (Benjamini-Yekutieli) ──────────────────────────────────
    combos_sorted = sorted(pvalores_raw, key=lambda c: pvalores_raw[c])
    m   = len(combos_sorted)
    c_m = sum(1 / k for k in range(1, m + 1))

    pval_bhy = {}
    for rank, combo in enumerate(combos_sorted, start=1):
        pval_bhy[combo] = min(pvalores_raw[combo] * m * c_m / rank, 1.0)

    # ── White's Reality Check (Hansen 2005 SPA) ───────────────────────────────
    # H₀: ninguna estrategia es genuinamente mejor que la media del grupo
    # Se impone H₀ centrando cada columna bootstrap en su propia media
    # V_obs = ventaja del ganador sobre la media observada
    # p_RC  = fracción de muestras bootstrap donde esa ventaja se supera por azar
    sh_boot_mean = sh_boot.mean(axis=0)
    sh_boot_c    = sh_boot - sh_boot_mean[np.newaxis, :]   # centrar bajo H₀
    V_obs        = float(sh_obs.max() - sh_obs.mean())     # ventaja observada del ganador
    V_boot       = sh_boot_c.max(axis=1)                   # max centrado por muestra
    p_rc         = float(np.mean(V_boot >= V_obs))
    print(f"White's Reality Check p-valor: {p_rc:.4f}  "
          f"({'NO significativo' if p_rc > ALPHA else 'Significativo al ' + str(int(ALPHA*100)) + '%'})\n")

    # ── Sensibilidad al tamaño de bloque ──────────────────────────────────────
    # Repite el bootstrap con bloques de 1, 3, 6 y 12 meses para comprobar
    # que los resultados no dependen críticamente del parámetro elegido.
    print("Sensibilidad al tamaño de bloque bootstrap:")
    print(f"  {'Bloque':>7}  {'Sharpe CI 95%':^22}  {'CAGR CI 95%':^22}  {'p-RC':>7}")
    print("  " + "-" * 67)
    rng_sens = np.random.default_rng(99)
    for bs in [1, 3, 6, 12]:
        n_blk = int(np.ceil(T / bs))
        sh_b  = np.zeros((N_BOOT, len(cols)))
        cagr_b = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            starts = rng_sens.integers(0, T, size=n_blk)
            idx_b  = np.concatenate([np.arange(s, s + bs) % T for s in starts])[:T]
            samp   = mat[idx_b]
            sh_b[b] = [_sharpe_anual(samp[:, k]) for k in range(len(cols))]
            cagr_b[b] = _cagr_anual(samp[:, idx_target])
        sh_lo, sh_hi     = np.nanpercentile(sh_b[:, idx_target], [2.5, 97.5])
        cagr_lo, cagr_hi = np.nanpercentile(cagr_b, [2.5, 97.5])
        # White's RC bajo H₀ centrado
        sh_b_c  = sh_b - sh_b.mean(axis=0)[np.newaxis, :]
        sh_obs_b = np.array([_sharpe_anual(mat[:, k]) for k in range(len(cols))])
        v_obs_b  = float(sh_obs_b.max() - sh_obs_b.mean())
        p_rc_b   = float(np.mean(sh_b_c.max(axis=1) >= v_obs_b))
        marker = " ◄" if bs == BLOCK_SIZE else ""
        print(f"  {bs:>4}M    [{sh_lo:+.3f}, {sh_hi:+.3f}]          "
              f"[{cagr_lo:+.1%}, {cagr_hi:+.1%}]      {p_rc_b:.4f}{marker}")
    print(f"  (◄ = bloque base del análisis principal)\n")

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    print("=" * 78)
    print(f"TEST PAREADO: {TARGET[0]}/{TARGET[1]} vs cada rival  "
          f"(H0: Sharpe iguales, test unilateral — target supera al rival)")
    print(f"{'Top 20 rivales mas cercanos':^78}")
    print("=" * 78)
    print(f"  {'Rival':<14} {'Sharpe obs':>11} {'p-val raw':>10} {'p-val BHY':>10} "
          f"{'Sig. BHY':>9} {'Concl.'}")
    print("  " + "-" * 72)

    rivales_df = pd.DataFrame({
        "sharpe": {c: sh_obs[i] for i, c in enumerate(cols) if c != TARGET},
        "p_raw":  pvalores_raw,
        "p_bhy":  pval_bhy,
    }).sort_values("p_raw")

    for combo, row in rivales_df.head(20).iterrows():
        sig   = "***" if row.p_bhy < 0.01 else ("**" if row.p_bhy < 0.05 else ("*" if row.p_bhy < 0.10 else "ns"))
        concl = "Target MEJOR" if row.p_bhy < ALPHA else "No concluyente"
        print(f"  {combo[0]+'/'+ combo[1]:<14} {row.sharpe:>11.3f} {row.p_raw:>10.4f} "
              f"{row.p_bhy:>10.4f} {sig:>9}  {concl}")

    print("=" * 78)
    print(f"\n  *** p<0.01  ** p<0.05  * p<0.10  ns=no significativo (BHY corregido)")
    print(f"  Sharpe observado target ({TARGET[0]}/{TARGET[1]}): {sh_target_obs:.3f}  "
          f"IC95%: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  White's Reality Check: p = {p_rc:.4f}")
    print("=" * 78)

    # ── Grafico: Heatmap p-valores BHY ────────────────────────────────────────
    FREQ_LABELS_HM = [fl for fl, _ in FRECUENCIAS if fl != "1M"]
    grid_pval = pd.DataFrame(np.nan, index=FREQ_LABELS_HM, columns=TOP_LABELS)

    for combo, pval in pval_bhy.items():
        fl, tl = combo
        if fl in grid_pval.index and tl in grid_pval.columns:
            grid_pval.loc[fl, tl] = pval

    fig, ax = plt.subplots(figsize=(13, 6))
    vals = grid_pval.values.astype(float)
    im   = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(TOP_LABELS)))
    ax.set_xticklabels(TOP_LABELS, fontsize=10)
    ax.set_yticks(range(len(FREQ_LABELS_HM)))
    ax.set_yticklabels(FREQ_LABELS_HM, fontsize=10)
    ax.set_xlabel("Top N%", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title(
        f"p-valor BHY corregido — H0: rival ≥ {TARGET[0]}/{TARGET[1]}\n"
        f"Verde oscuro = rival claramente inferior  |  Rojo = no concluyente",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label="p-valor BHY")
    for i in range(len(FREQ_LABELS_HM)):
        for j in range(len(TOP_LABELS)):
            val = float(grid_pval.iloc[i, j])
            if not np.isnan(val):
                sig = "***" if val < 0.01 else ("**" if val < 0.05 else ("*" if val < 0.10 else ""))
                tc  = "white" if val < 0.08 else "black"
                ax.text(j, i, f"{val:.2f}\n{sig}", ha="center", va="center",
                        fontsize=7.5, color=tc, fontweight="bold")

    plt.tight_layout()
    ruta2 = OUTPUTS / "hypothesis_pvalues.png"
    plt.savefig(ruta2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafico guardado: {ruta2}")
