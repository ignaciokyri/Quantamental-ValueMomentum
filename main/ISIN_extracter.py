import pdfplumber
import os
import re
from pathlib import Path

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
CARPETA_PDFS   = str(BASE_DIR.parent / "data")
CARPETA_SALIDA = str(BASE_DIR.parent / "outputs")

# Patrón ISIN: exactamente 2 letras mayúsculas seguidas de 10 caracteres alfanuméricos.
# Acepta tanto " - ACCIONES|" como "-ACCIONES " como separadores (variantes reales del PDF).
# Grupos de captura: (1) ISIN  (2) nombre de la compañía
PATRON_ISIN = re.compile(
    r"([A-Z]{2}[A-Z0-9]{10})"   # grupo 1: ISIN
    r"\s*-?\s*ACCIONES[\s|]*"   # separador flexible
    r"(.+)",                     # grupo 2: nombre (resto de la cadena)
    re.IGNORECASE,
)


def extraer_empresas(ruta_pdf: str) -> dict[str, str]:
    """Abre un PDF y devuelve {ISIN: nombre} con todas las empresas encontradas."""

    # Usamos un dict {ISIN: nombre} para deduplicar automáticamente.
    # Si el mismo ISIN aparece en varias tablas, la última escritura prevalece.
    empresas: dict[str, str] = {}

    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            tablas = pagina.extract_tables()

            if not tablas:
                continue  # página sin tablas estructuradas → siguiente

            for tabla in tablas:
                for fila in tabla:
                    for celda in fila:
                        if not celda:
                            continue  # celda vacía o None

                        # Limpieza: el extractor de PDF a veces rompe una misma línea
                        # lógica en varias líneas físicas (\n / \r). Reemplazamos esos
                        # saltos por un espacio para que el patrón los cruce sin problemas.
                        celda_limpia = celda.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

                        # Comprimimos espacios múltiples que puedan haber quedado.
                        celda_limpia = re.sub(r"\s{2,}", " ", celda_limpia).strip()

                        # Aplicamos el patrón sobre la celda ya limpia.
                        coincidencia = PATRON_ISIN.search(celda_limpia)
                        if coincidencia:
                            isin   = coincidencia.group(1).upper()
                            nombre = coincidencia.group(2).strip()

                            # Eliminamos posibles artefactos al final del nombre
                            # (caracteres de control u otros separadores residuales).
                            nombre = re.sub(r"[\|\t]+.*$", "", nombre).strip()

                            empresas[isin] = nombre

    return empresas


# Patrón para extraer año y semestre del nombre del archivo (fallback).
PATRON_ARCHIVO = re.compile(r"(\d{4})-Semestre\s*(\d)", re.IGNORECASE)

# Patrón principal: "Informe Semestral del Primer/Segundo Semestre YYYY"
# Formato estándar CNMV — captura semestre y año en un único match.
_PATRON_PERIODO = re.compile(
    r"Informe Semestral del (Primer|Segundo) Semestre\s+(20\d{2})",
    re.IGNORECASE,
)


def extraer_periodo_del_pdf(ruta_pdf: str) -> tuple[str, str] | None:
    """
    Lee las primeras páginas del PDF e intenta detectar año y semestre.
    Devuelve (año, semestre) como strings, o None si no lo encuentra.
    """
    with pdfplumber.open(ruta_pdf) as pdf:
        texto = " ".join(p.extract_text() or "" for p in pdf.pages[:4])

    m = _PATRON_PERIODO.search(texto)
    if m:
        semestre = "1" if m.group(1).lower() == "primer" else "2"
        return m.group(2), semestre
    return None

# ─── Bucle principal: procesa todos los PDFs de la carpeta ───────────────────
archivos_pdf = sorted(
    f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith(".pdf")
)

if not archivos_pdf:
    print(f"No se encontraron archivos PDF en:\n  {CARPETA_PDFS}")
else:
    # Lista de tuplas (isin, nombre, año, semestre).
    # Una misma empresa puede aparecer en varios informes, por eso usamos lista
    # en lugar de dict: queremos una fila por cada (empresa, informe).
    registros: list[tuple[str, str, str, str]] = []

    # Set para deduplicar dentro del mismo informe: (isin, año, semestre).
    vistos: set[tuple[str, str, str]] = set()

    for nombre_archivo in archivos_pdf:
        ruta_pdf = os.path.join(CARPETA_PDFS, nombre_archivo)

        # Intentar extraer año y semestre del contenido del PDF.
        periodo = extraer_periodo_del_pdf(ruta_pdf)
        if periodo:
            año, semestre = periodo
        else:
            # Fallback: intentar extraerlo del nombre del archivo.
            m = PATRON_ARCHIVO.search(nombre_archivo)
            if not m:
                print(f"Saltando '{nombre_archivo}': no se pudo determinar año/semestre")
                continue
            año, semestre = m.group(1), m.group(2)
            print(f"  (año/semestre extraído del nombre de archivo)")

        print(f"Procesando: {nombre_archivo} ...", end=" ", flush=True)

        empresas = extraer_empresas(ruta_pdf)

        nuevas = 0
        for isin, nombre in empresas.items():
            clave = (isin, año, semestre)
            if clave not in vistos:
                vistos.add(clave)
                registros.append((isin, nombre, año, semestre))
                nuevas += 1

        print(f"{nuevas} empresas encontradas")

    # Ordenamos por año → semestre → ISIN para facilitar la lectura.
    registros.sort(key=lambda r: (r[2], r[3], r[0]))

    # ─── Escritura del fichero consolidado ───────────────────────────────────
    ruta_salida = os.path.join(CARPETA_SALIDA, "companies.txt")

    with open(ruta_salida, "w", encoding="utf-8") as f:
        for isin, nombre, año, semestre in registros:
            f.write(f"{isin}, {nombre}, {año}, Semestre {semestre}\n")

    print(f"\nTotal: {len(registros)} registros → companias_cobas.txt")
