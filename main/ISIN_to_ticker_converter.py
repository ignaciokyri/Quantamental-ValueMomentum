import re
import requests
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent   # raiz del proyecto
OUTPUTS_DIR  = BASE_DIR / "outputs"
RUTA_ENTRADA = OUTPUTS_DIR / "companies.txt"
RUTA_SALIDA  = OUTPUTS_DIR / "tickers.txt"

# Yahoo Finance bloquea peticiones sin User-Agent reconocible.
# Simulamos un navegador Chrome estándar para evitar respuestas 429 / vacías.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Pausa entre peticiones (segundos) para no saturar la API de Yahoo.
PAUSA = 0.5


def buscar_ticker(isin: str) -> str | None:
    """
    Consulta el endpoint de búsqueda de Yahoo Finance con el ISIN dado.

    Respuesta JSON esperada (estructura simplificada):
    {
      "quotes": [
        { "symbol": "RNO.PA", "shortname": "Renault", ... },
        ...
      ]
    }
    El primer resultado ('quotes[0]') suele ser la coincidencia más relevante.
    Devuelve el 'symbol' si existe, o None si no hay resultados o falla la petición.
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={isin}"

    try:
        respuesta = requests.get(url, headers=HEADERS, timeout=10)
        respuesta.raise_for_status()

        quotes = respuesta.json().get("quotes", [])
        if quotes:
            return quotes[0].get("symbol")

    except requests.RequestException as e:
        print(f"  Error de red para {isin}: {e}")

    return None


# ─── Lectura de companias_cobas.txt ──────────────────────────────────────────
# Formato actual de cada línea: "ISIN, Nombre, Año, Semestre N"
with open(RUTA_ENTRADA, "r", encoding="utf-8") as f:
    lineas = [l.strip() for l in f if l.strip()]

# ─── Conversión ISIN → Ticker con caché ──────────────────────────────────────
# El mismo ISIN puede aparecer en múltiples semestres.
# Guardamos en caché los resultados ya consultados para no repetir llamadas a Yahoo.
cache_isin: dict[str, str | None] = {}

resultados: list[tuple[str, str, str, str]] = []  # [(ticker, nombre, año, semestre)]
mapeadas = 0

for linea in lineas:
    # Separamos en exactamente 4 partes: ISIN, nombre, año, semestre.
    # maxsplit=3 preserva posibles comas dentro del nombre de empresa.
    partes = [p.strip() for p in linea.split(",", maxsplit=3)]
    if len(partes) != 4:
        continue

    isin, nombre, año, semestre = partes

    # Consultamos Yahoo solo si no hemos procesado este ISIN antes.
    if isin not in cache_isin:
        print(f"Buscando {isin} ({nombre}) ...", end=" ", flush=True)
        ticker = buscar_ticker(isin)
        cache_isin[isin] = ticker

        if ticker:
            print(f"→ {ticker}")
        else:
            print(f"→ No se pudo mapear el ISIN: {isin}")

        time.sleep(PAUSA)
    else:
        ticker = cache_isin[isin]

    if ticker:
        resultados.append((ticker, nombre, año, semestre))
        mapeadas += 1

# ─── Escritura de tickers_cobas.txt ──────────────────────────────────────────
with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
    for ticker, nombre, año, semestre in resultados:
        f.write(f"{ticker}, {nombre}, {año}, {semestre}\n")

# ─── Resumen ──────────────────────────────────────────────────────────────────
n_unicos    = len(cache_isin)
n_ok        = sum(1 for t in cache_isin.values() if t)
n_fail      = n_unicos - n_ok
pct_ok      = n_ok   / n_unicos * 100 if n_unicos else 0
pct_fail    = n_fail / n_unicos * 100 if n_unicos else 0

# Nombre de cada ISIN fallido (tomamos el último nombre visto en companies.txt)
nombre_por_isin: dict[str, str] = {}
for linea in lineas:
    partes = [p.strip() for p in linea.split(",", maxsplit=3)]
    if len(partes) == 4:
        nombre_por_isin[partes[0]] = partes[1]

print(f"\n{'═'*55}")
print(f"  ISINs únicos consultados : {n_unicos}")
print(f"  Mapeados con ticker      : {n_ok:>3}  ({pct_ok:.1f}%)")
print(f"  Sin ticker               : {n_fail:>3}  ({pct_fail:.1f}%)")
print(f"{'═'*55}")

if n_fail:
    print(f"\n  ISINs sin ticker ({n_fail}):")
    for isin, ticker in cache_isin.items():
        if not ticker:
            nombre = nombre_por_isin.get(isin, "—")
            print(f"    {isin}  {nombre}")

print(f"\n{mapeadas} registros escritos → {RUTA_SALIDA.name}")

# ─── Gráfico por semestre ──────────────────────────────────────────────────────
# Contar ISINs únicos por semestre y cuántos se mapearon
from collections import defaultdict

sem_total: dict[tuple, set] = defaultdict(set)
sem_ok:    dict[tuple, set] = defaultdict(set)

for linea in lineas:
    partes = [p.strip() for p in linea.split(",", maxsplit=3)]
    if len(partes) != 4:
        continue
    isin, _, año_str, sem_str = partes
    try:
        año = int(año_str)
    except ValueError:
        continue
    m = re.search(r"(\d)", sem_str)
    if not m:
        continue
    sem = int(m.group(1))
    sem_total[(año, sem)].add(isin)
    if cache_isin.get(isin):
        sem_ok[(año, sem)].add(isin)

periodos  = sorted(sem_total.keys())
labels    = [f"{a} S{s}" for a, s in periodos]
totales   = np.array([len(sem_total[p]) for p in periodos])
mapeados  = np.array([len(sem_ok[p])    for p in periodos])
fallidos  = totales - mapeados
pct_fallo = np.where(totales > 0, fallidos / totales * 100, 0.0)

fig, ax = plt.subplots(figsize=(max(10, len(periodos) * 0.9), 6))

x = np.arange(len(periodos))
w = 0.6

bars_ok  = ax.bar(x, mapeados, width=w, color="#27ae60", label="Mapeados (con ticker)")
bars_ko  = ax.bar(x, fallidos, width=w, bottom=mapeados,
                  color="#e74c3c", label="Sin ticker", alpha=0.85)

# Anotar % fallido encima de cada barra
for i, (tot, pct) in enumerate(zip(totales, pct_fallo)):
    if tot == 0:
        continue
    ax.text(x[i], tot + 0.4, f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=8.5,
            color="#c0392b" if pct > 0 else "#27ae60", fontweight="bold")

# Línea de % fallido sobre eje secundario
ax2 = ax.twinx()
ax2.plot(x, pct_fallo, color="#c0392b", linewidth=1.6,
         linestyle="--", marker="o", markersize=5, label="% sin ticker")
ax2.set_ylabel("% sin ticker", fontsize=10, color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_ylim(0, max(pct_fallo.max() * 1.4, 10))

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("N.º de empresas únicas", fontsize=10)
ax.set_title(
    f"Universo COBAS por semestre — conversión ISIN → Ticker\n"
    f"Total ISINs únicos: {n_unicos}  |  Mapeados: {n_ok} ({pct_ok:.1f}%)  |  "
    f"Sin ticker: {n_fail} ({pct_fail:.1f}%)",
    fontsize=11, fontweight="bold",
)
ax.legend(loc="upper left", fontsize=9)
ax2.legend(loc="upper right", fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
ruta_png = OUTPUTS_DIR / "isin_coverage.png"
plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gráfico guardado: {ruta_png}")
