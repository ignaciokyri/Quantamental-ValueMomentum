# Guía Detallada de Obtención de Datos y Ejecución

Este documento proporciona instrucciones paso a paso para obtener manualmente los conjuntos de datos oficiales requeridos por la estrategia, así como la secuencia de comandos para ejecutar el pipeline analítico completo. Todos los datos necesarios son completamente gratuitos y provienen íntegramente de plataformas regulatorias y financieras públicas y transparentes.

---

## 1. Obtención de los Datos Oficiales

Debido a restricciones de copyright y redistribución, para replicar el universo de inversión y el rendimiento histórico del fondo, es necesario descargar manualmente dos bloques de datos distintos:

### A. Composición de la Cartera (Informes Semestrales CNMV)
Los datos de asignación de activos del fondo se extraen de los informes regulatorios oficiales. Puedes acceder a ellos de dos formas:

#### Opción 1: Acceso Directo (Recomendado)
1. Accede directamente al perfil oficial del fondo mediante este enlace: [CNMV - Registro de Cobas Selección FI](https://www.cnmv.es/portal/consultas/iic/fondo?nif=V87647368&vista=1)
2. Una vez en la página, pasa directamente al **Paso 8** a continuación.

#### Opción 2: Navegación Manual Paso a Paso
1. Accede al portal oficial de la **[CNMV (Comisión Nacional del Mercado de Valores)](https://www.cnmv.es/)**.
2. En el menú lateral, haz clic en **"Consultas a registros oficiales"**.
3. Selecciona la opción **"IIC nacionales, gestoras y depositarios"**.
4. Dentro de esta sección, haz clic en **"Listado de entidades"**.
5. Selecciona **"Listado completo de fondos de inversión de carácter financiero o FI armonizados"**.
6. Navega hasta la **Página 44** (según la estructura actual del registro).
7. Busca y selecciona **"COBAS SELECCION, FI"**.
8. Dentro del registro oficial del fondo, navega a la pestaña **"Información pública periódica"**.
9. Descarga únicamente los **informes semestrales** desde 2017 hasta el semestre más reciente cerrado (da igual el nombre que les pongas).
10. Coloca todos los archivos PDF descargados en el directorio del proyecto bajo en la ruta `data/`.

### B. Rendimiento Histórico del Fondo (Cobas Asset Management)
Para comparar la estrategia cuantitativa con el rendimiento real del fondo, necesitas su Valor Liquidativo (VL) histórico. Puedes acceder a él de dos formas:

#### Opción 1: Acceso Directo (Recomendado)
1. Accede directamente a la página oficial del producto mediante este enlace: [Cobas Asset Management - Cobas Selección FI - Clase C](https://www.cobasam.com/productos/inversion-libre/cobas_seleccion/#COBAS_SELECCION_C)
2. Una vez en la página, pasa directamente al **Paso 6** a continuación.

#### Opción 2: Navegación Manual Paso a Paso
1. Accede al portal oficial de **[Cobas Asset Management](https://www.cobasam.com/)**.
2. En el menú de navegación superior, pasa el cursor o haz clic en **"Productos"** y selecciona **"Fondos de inversión"**.
3. Localiza el fondo **"Cobas Selección FI"**.
4. Haz clic en **"Ver detalles"** de la clase C.
5. Dentro de los detalles del fondo, selecciona la pestaña correspondiente a **"Clase C"**.
6. Localiza la sección **"HISTÓRICO VALORES LIQUIDATIVOS"** para descargar el archivo.
7. Coloca el archivo dentro de:
    `data/`

---

## 2. Pipeline de Ejecución del Código

Una vez que los archivos mencionados estén en sus carpetas locales correspondientes, para iniciar el procesamiento de datos ejecuta el script `main.py`.