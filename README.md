# Estimación del salario en España con la Encuesta de Estructura Salarial

Trabajo Fin de Máster · Máster en Data Science, Big Data & Business Analytics (UCM)
Autor: Alfredo Brocal Serrano

Modelo que estima el rango de salario bruto anual (percentiles 10, 50 y 90) de un trabajador a partir de su perfil laboral, usando los microdatos de la Encuesta de Estructura Salarial del INE (ediciones 2018 y 2022, 457.216 registros).

## Estructura

```
data/raw/          microdatos originales del INE (no incluidos en el repositorio)
data/processed/    datos preparados que generan los notebooks
notebooks/         análisis paso a paso (01 carga, 02 EDA, ...)
src/               funciones reutilizables (las usa también la API)
api/               API de predicción (FastAPI)
web/               aplicación web
reports/figures/   gráficos para la memoria
```

## Cómo reproducirlo

1. Descargar los microdatos de la Encuesta Cuatrienal de Estructura Salarial desde el INE
   (INEbase → Salarios y costes laborales → Encuestas de estructura salarial → Microdatos):
   https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177025&menu=resultados&idp=1254735976596
   - Año 2022 → Microdatos
   - Año 2018 → Microdatos
2. Copiar en `data/raw/`:
   - `md_EES_2018.parquet`, `md_EES_2022.parquet`
   - `dr_EES_2018.xlsx`, `dr_EES_2022.xlsx` (diseño de registro)
3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Ejecutar los notebooks en orden desde la carpeta `notebooks/`.

## Fuente de los datos

Fuente: INE, www.ine.es. Encuesta Cuatrienal de Estructura Salarial 2018 y 2022.
Los resultados y cálculos derivados son responsabilidad exclusiva del autor.
