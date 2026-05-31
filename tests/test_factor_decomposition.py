"""
Descomposicion factorial — Fama-French 6 factores (Developed Markets ex US)

Cuestion central
────────────────
La estrategia 12-1 combina un universo value (COBAS/Parames) con un filtro
momentum 12-1. La regresion factorial responde:

  (a) HML (valor)  : el rendimiento proviene de comprar acciones baratas?
  (b) WML (moment.): el rendimiento proviene de comprar ganadoras recientes?
  (c) Alpha        : queda algo inexplicable por factores estandar?

Modelo estimado (OLS con errores Newey-West, lag=3):
  r_t - RF_t = alpha + b_MKT*(MKT-RF) + b_SMB*SMB + b_HML*HML
                     + b_RMW*RMW + b_CMA*CMA + b_WML*WML + e_t

Factores: Fama-French Developed ex US (mensual, desde K. French Data Library)

Outputs
───────
  outputs/factor_decomposition.png
  Tabla consola: alpha, betas, t-stats NW, p-valores, R2
"""

import importlib.util
import io
import re
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"

_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


# ── Descarga y parseo de factores Fama-French ─────────────────────────────────

def _parsear_ff_csv(raw: str, col_names: list) -> pd.DataFrame:
    """
    Extrae filas mensuales (YYYYMM) de un CSV de K. French.
    Los valores estan en porcentaje; los divide por 100.
    """
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not re.match(r"^\d{6}$", parts[0]):
            continue
        if len(parts) < len(col_names) + 1:
            continue
        try:
            date = pd.to_datetime(parts[0], format="%Y%m") + pd.offsets.MonthEnd(0)
            vals = [float(parts[i + 1]) / 100.0 for i in range(len(col_names))]
            rows.append([date] + vals)
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows, columns=["date"] + col_names).set_index("date")


def _descargar_zip_csv(nombre: str) -> str:
    url  = f"{_FF_BASE}/{nombre}_CSV.zip"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def descargar_factores() -> pd.DataFrame:
    """Descarga y combina 5 factores + momentum (Developed ex US)."""
    print("  Descargando 5 factores (Developed ex US)...")
    raw5   = _descargar_zip_csv("Developed_ex_US_5_Factors")
    ff5    = _parsear_ff_csv(raw5, ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"])

    print("  Descargando factor momentum (Developed ex US)...")
    raw_m  = _descargar_zip_csv("Developed_ex_US_Mom_Factor")
    mom    = _parsear_ff_csv(raw_m, ["WML"])

    df = ff5.join(mom, how="left")
    print(f"  Factores: {len(df)} meses ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# ── Carga del backtest ────────────────────────────────────────────────────────

def _cargar_bt():
    spec = importlib.util.spec_from_file_location("bt", BASE / "main" / "backtest.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.stdout = io.StringIO()
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = sys.__stdout__
    return mod


# ── OLS con errores Newey-West ────────────────────────────────────────────────

def _ols_nw(y: np.ndarray, X: np.ndarray, lag: int = 3) -> dict:
    """
    OLS con varianza Newey-West de lag meses.
    Devuelve alpha mensual/anual, betas, t-stats, p-valores, R2.
    """
    n, k = X.shape
    Xc   = np.column_stack([np.ones(n), X])   # añadir constante

    coeffs, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ coeffs

    # R²
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if ss_tot > 0 else np.nan

    # Matriz de varianza Newey-West
    XtXinv = np.linalg.pinv(Xc.T @ Xc)
    S      = sum(resid[t] ** 2 * np.outer(Xc[t], Xc[t]) for t in range(n))
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        for t in range(l, n):
            outer = resid[t] * resid[t - l] * (
                np.outer(Xc[t], Xc[t - l]) + np.outer(Xc[t - l], Xc[t])
            )
            S += w * outer
    vcov   = XtXinv @ S @ XtXinv
    se     = np.sqrt(np.maximum(np.diag(vcov), 0))
    tstats = coeffs / (se + 1e-12)

    # p-valores (aproximacion normal; con n>60 es adecuado)
    pvalues = 2 * (1 - _norm_cdf(np.abs(tstats)))

    return {
        "alpha_m":  float(coeffs[0]),
        "alpha_a":  float((1 + coeffs[0]) ** 12 - 1),
        "betas":    coeffs[1:],
        "tstat_a":  float(tstats[0]),
        "tstats":   tstats[1:],
        "pvalues":  pvalues,
        "r2":       r2,
        "r2_adj":   r2_adj,
        "resid":    resid,
        "n":        n,
    }


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(lambda v: float(
        1 - 2 * sum((-1)**k * v**(2*k+1) / (2**k * __import__('math').factorial(k) * (2*k+1))
                    for k in range(50)) / __import__('math').sqrt(2 * __import__('math').pi)
        if abs(v) < 6 else (1.0 if v > 0 else 0.0)
    ))(x))


# ── Funcion auxiliar para p-valores via scipy (si disponible) ─────────────────

def _pvalues_t(tstats: np.ndarray, df: int) -> np.ndarray:
    try:
        from scipy import stats
        return np.array([2 * (1 - stats.t.cdf(abs(t), df)) for t in tstats])
    except ImportError:
        return _norm_cdf_pval(tstats)


def _norm_cdf_pval(tstats: np.ndarray) -> np.ndarray:
    """p-valor aproximado via formula de Abramowitz & Stegun."""
    import math
    result = []
    for t in tstats:
        x = abs(t) / math.sqrt(2)
        # erf approx
        a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
        p_erf = 0.3275911
        sign  = 1.0
        tt = 1.0 / (1.0 + p_erf * x)
        y  = 1.0 - (((((a5*tt + a4)*tt) + a3)*tt + a2)*tt + a1)*tt * math.exp(-x*x)
        result.append(2 * (1 - (1 + sign * y) / 2))
    return np.array(result)


def _ols_clean(y: np.ndarray, X: np.ndarray, lag: int = 3) -> dict:
    """OLS con errores HAC via statsmodels si disponible, sino NW manual."""
    try:
        import statsmodels.api as sm
        model  = sm.OLS(y, sm.add_constant(X))
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        n, k   = X.shape
        y_hat  = result.fittedvalues
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2     = 1 - ss_res / ss_tot
        r2_adj = result.rsquared_adj
        return {
            "alpha_m":  float(result.params[0]),
            "alpha_a":  float((1 + result.params[0]) ** 12 - 1),
            "betas":    result.params[1:],
            "tstat_a":  float(result.tvalues[0]),
            "tstats":   result.tvalues[1:],
            "pvalues":  result.pvalues,
            "r2":       r2,
            "r2_adj":   r2_adj,
            "resid":    result.resid,
            "n":        n,
        }
    except ImportError:
        res = _ols_nw(y, X, lag)
        n   = res["n"]
        k   = X.shape[1]
        all_t = np.concatenate([[res["tstat_a"]], res["tstats"]])
        res["pvalues"] = _pvalues_t(all_t, df=n - k - 1)
        return res


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Retornos mensuales de la estrategia ────────────────────────────────
    bt = _cargar_bt()
    print("Cargando universo y precios...")
    universo = bt.cargar_universo(bt.RUTA_TICKERS)
    precios  = bt.descargar_precios(universo)

    print("Ejecutando backtest 1M/15%...")
    _old_stdout = sys.stdout
    sys.stdout  = io.StringIO()
    equity      = bt.ejecutar_backtest_mensual(universo, precios)
    sys.stdout  = _old_stdout

    ret_m = equity.resample("ME").last().pct_change().dropna()
    print(f"Retornos mensuales: {len(ret_m)} obs  "
          f"({ret_m.index[0].date()} -> {ret_m.index[-1].date()})\n")

    # ── 2. Factores Fama-French ───────────────────────────────────────────────
    print("Descargando factores Fama-French...")
    factores = descargar_factores()

    # ── 3. Alinear ────────────────────────────────────────────────────────────
    df = factores.copy()
    df["estrategia"] = ret_m
    df = df.dropna()
    print(f"\nMeses con datos completos: {len(df)}  "
          f"({df.index[0].date()} -> {df.index[-1].date()})\n")

    df["r_exc"] = df["estrategia"] - df["RF"]

    FACTORES = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "WML"]
    LABELS   = {
        "MKT_RF": "MKT-RF  (mercado)",
        "SMB":    "SMB     (tamanio)",
        "HML":    "HML     (valor)  ",
        "RMW":    "RMW     (profit.)",
        "CMA":    "CMA     (invers.)",
        "WML":    "WML     (moment.)",
    }
    factor_cols = [f for f in FACTORES if f in df.columns]

    y = df["r_exc"].values
    X = df[factor_cols].values

    res = _ols_clean(y, X)

    # ── 4. Tabla de resultados ────────────────────────────────────────────────
    def sig(p): return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))

    print("=" * 70)
    print("DESCOMPOSICION FACTORIAL  —  Fama-French 6F  (Developed ex US)")
    print(f"Muestra: {df.index[0].date()} - {df.index[-1].date()}  "
          f"(n={res['n']} meses)")
    print("=" * 70)
    print(f"  {'Variable':<22} {'Coef':>10} {'t-stat NW':>10} {'p-valor':>9}  Sig.")
    print("  " + "-" * 58)

    print(f"  {'Alpha (mensual)':22} {res['alpha_m']:>+10.4f} "
          f"{res['tstat_a']:>10.2f} {res['pvalues'][0]:>9.4f}  {sig(res['pvalues'][0])}")
    print(f"  {'Alpha (anual)':22} {res['alpha_a']:>+10.2%}")
    print("  " + "-" * 58)

    for i, fc in enumerate(factor_cols):
        label = LABELS.get(fc, fc)
        print(f"  {label:22} {res['betas'][i]:>+10.4f} "
              f"{res['tstats'][i]:>10.2f} {res['pvalues'][i+1]:>9.4f}  {sig(res['pvalues'][i+1])}")

    print("  " + "-" * 58)
    print(f"  R2:                    {res['r2']:>10.4f}")
    print(f"  R2 ajustado:           {res['r2_adj']:>10.4f}")
    print("=" * 70)

    # ── 5. Interpretacion ────────────────────────────────────────────────────
    print("\nInterpretacion:")
    print(f"  Alpha anual {res['alpha_a']:+.1%} (t={res['tstat_a']:.2f}, "
          f"p={res['pvalues'][0]:.3f}): "
          + ("SIGNIFICATIVO al 10%" if res['pvalues'][0] < 0.10
             else "no significativo — compatible con factores puros"))

    hml_i = factor_cols.index("HML")
    print(f"  HML (valor)  beta={res['betas'][hml_i]:+.3f}  "
          + ("exposicion value significativa -> universo COBAS aporta tilt value"
             if res['pvalues'][hml_i+1] < 0.10 and res['betas'][hml_i] > 0
             else "no significativo al 10%"))

    if "WML" in factor_cols:
        wml_i = factor_cols.index("WML")
        print(f"  WML (moment) beta={res['betas'][wml_i]:+.3f}  "
              + ("exposicion momentum significativa -> filtro 12-1 aporta tilt momentum"
                 if res['pvalues'][wml_i+1] < 0.10 and res['betas'][wml_i] > 0
                 else "no significativo al 10%"))

    print(f"  R2={res['r2']:.2f}: los factores explican el {res['r2']:.0%} de la varianza mensual")
    unexplained = 1 - res["r2"]
    print(f"  El {unexplained:.0%} restante es idiosincratico (alpha + ruido no capturado por 6F)")

    # ── 6. Alpha rodante (ventana = 24 meses) ────────────────────────────────
    WINDOW = 24
    roll_alpha, roll_dates = [], []
    for i in range(WINDOW, len(df) + 1):
        chunk = df.iloc[i - WINDOW:i]
        yr    = chunk["r_exc"].values
        Xr    = chunk[factor_cols].values
        Xrc   = np.column_stack([np.ones(WINDOW), Xr])
        try:
            c, _, _, _ = np.linalg.lstsq(Xrc, yr, rcond=None)
            roll_alpha.append(float((1 + c[0]) ** 12 - 1))
        except Exception:
            roll_alpha.append(np.nan)
        roll_dates.append(chunk.index[-1])

    # ── 7. Grafico ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Panel 1: alpha rodante
    ax1 = axes[0]
    arr = np.array(roll_alpha) * 100
    col_bar = ["#27ae60" if v >= 0 else "#c0392b" for v in arr]
    ax1.bar(roll_dates, arr, color=col_bar, width=25, alpha=0.80)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(res["alpha_a"] * 100, color="#8e44ad", linewidth=1.6,
                linestyle="--", label=f"Alpha total: {res['alpha_a']:+.1%}")
    ax1.set_title(f"Alpha rodante ({WINDOW}M ventana) — modelo Fama-French 6F",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Alpha anualizado (%)", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.grid(True, axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Panel 2: betas con t-stats anotados
    ax2 = axes[1]
    betas_arr = res["betas"]
    col_beta  = ["#2980b9" if b >= 0 else "#c0392b" for b in betas_arr]
    bars      = ax2.bar(range(len(betas_arr)), betas_arr, color=col_beta, alpha=0.80, width=0.55)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(range(len(factor_cols)))
    ax2.set_xticklabels(factor_cols, fontsize=11)
    ax2.set_ylabel("Beta", fontsize=10)
    ax2.set_title("Exposicion factorial — Betas OLS (t-stats Newey-West, lag=3)",
                  fontsize=12, fontweight="bold")
    for j, (bar, ts, pv) in enumerate(zip(bars, res["tstats"], res["pvalues"][1:])):
        h    = float(betas_arr[j])
        mark = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
        ypos = h + 0.03 if h >= 0 else h - 0.06
        va   = "bottom" if h >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"t={ts:.1f}{mark}", ha="center", va=va, fontsize=9.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Estrategia 12-1 (1M / Top 15%) — Descomposicion Fama-French 6F",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    ruta = OUTPUTS / "factor_decomposition.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafico guardado: {ruta}")
