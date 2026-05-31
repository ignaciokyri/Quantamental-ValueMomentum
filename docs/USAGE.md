# Detailed Data Acquisition & Execution Guide

This document provides step-by-step instructions on how to manually source the official datasets required by the strategy, as well as the sequential command structure to run the entire analytical pipeline. All required underlying data is completely free of charge and sourced entirely from public, transparent regulatory and financial platforms.

---

## 1. Sourcing the Official Data

To replicate the investment universe and the fund's historical performance, you must manually download two distinct data blocks due to copyright and redistribution restrictions:

### A. Portfolio Constituents (CNMV Semi-Annual Reports)
The asset allocation data for the fund is extracted from official regulated regulatory disclosures. You can access it in two ways:

#### Option 1: Direct Shortcut (Recommended)
1. Go directly to the official fund profile via this link: [CNMV - Cobas Selección FI Registry](https://www.cnmv.es/portal/consultas/iic/fondo?nif=V87647368&vista=1)
2. Once on the page, proceed directly to **Step 8** below.

#### Option 2: Step-by-Step Manual Navigation
1. Access the official portal of the **[CNMV (Spanish Securities Market Commission)](https://www.cnmv.es/)**.
2. On the sidebar menu, click on **"Consultas a registros oficiales"** (Queries to official registries).
3. Select the option **"IIC nacionales, gestoras y depositarios"** (National CIUs, management companies, and depositories).
4. Within this section, click on **"Listado de entidades"** (List of entities).
5. Select **"Listado completo de fondos de inversión de carácter financiero o FI armonizados"** (Complete list of financial or harmonised investment funds).
6. Navigate to **Page 44** (as of today's current registry structure).
7. Find and select **"COBAS SELECCION, FI"** (Official registration details: *Número y fecha de registro oficial: 5075 - 14/10/2016 | Gestora: COBAS ASSET MANAGEMENT, SGIIC, S.A. | Depositario: BNP PARIBAS S.A., SUCURSAL EN ESPAÑA*).
8. Once inside the fund's official record, navigate to the tab named **"Información pública periódica"** (Periodic Public Information).
9. Download exclusively the **semi-annual reports** (*informes semestrales*) from 2017 to the most recently closed semester (the filename doesn't matter).
10. Place all downloaded PDF files into the project directory under the following exact path:
    `data/`

### B. Fund Historical Performance (Cobas Asset Management)
To benchmark the quantitative strategy against the actual fund performance, you need its historical Net Asset Value (NAV). You can access it in two ways:

#### Option 1: Direct Shortcut (Recommended)
1. Go directly to the official product page via this link: [Cobas Asset Management - Cobas Selección FI - Clase C](https://www.cobasam.com/productos/inversion-libre/cobas_seleccion/#COBAS_SELECCION_C)
2. Once on the page, proceed directly to **Step 6** below.

#### Option 2: Step-by-Step Manual Navigation
1. Access the official portal of **[Cobas Asset Management](https://www.cobasam.com/)**.
2. On the top navigation menu, hover or click on **"Productos"** (Products) and select **"Fondos de inversión"** (Investment Funds).
3. Locate the fund named **"Cobas Selección FI"**.
4. Click on **"Ver detalles"** (View details) for the Clase C share class.
5. Once inside the fund details, select the tab corresponding to **"Clase C"** (Class C).
6. Locate the **"HISTÓRICO VALORES LIQUIDATIVOS"** (Historical NAVs) section and download the file.
7. Place the file inside:
    `data/`

---

## 2. Code Execution Pipeline

Once the files mentioned above are placed in their respective local folders, run `main.py` to execute the full pipeline.
