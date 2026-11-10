"""Carga y preparación de los microdatos de la Encuesta de Estructura Salarial (INE).

Fuente: INE, www.ine.es - Encuesta Cuatrienal de Estructura Salarial 2018 y 2022.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"

ANIOS = (2018, 2022)

# Salario bruto anual medio publicado por el INE (nota de prensa EES 2022 / EES 2018).
# Se usa para validar que la variable objetivo está bien construida.
MEDIA_PUBLICADA_INE = {2018: 24009.12, 2022: 26948.87}

COMPONENTES_SALARIO = ["RETRINOIN", "RETRIIN", "VESPNOIN", "VESPIN"]


def leer_ees(anio: int, carpeta: Path = DATA_RAW) -> pd.DataFrame:
    """Lee el fichero parquet de microdatos de un año y añade la columna `anio`."""
    ruta = carpeta / f"md_EES_{anio}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra {ruta}.\n"
            f"Descarga los microdatos de la Encuesta de Estructura Salarial {anio} desde el INE "
            f"(ver README, apartado 'Cómo reproducirlo') y copia md_EES_{anio}.parquet en {carpeta}"
        )
    df = pd.read_parquet(ruta)
    df["anio"] = anio
    return df


def dias_anio_computables(df: pd.DataFrame) -> pd.Series:
    """Días del año que computan para el salario.

    Días de alta (meses * 30,42 + días) menos los días en situaciones especiales
    DSIESPA2 y DSIESPA4, siguiendo el documento del INE
    "Obtención de los resultados publicados".
    """
    return df["DRELABAM"] * 30.42 + df["DRELABAD"] - df["DSIESPA2"] - df["DSIESPA4"]


def calcular_salario_anual(df: pd.DataFrame) -> pd.Series:
    """Salario bruto anual elevado a año completo (365 días)."""
    bruto = df[COMPONENTES_SALARIO].sum(axis=1)
    return bruto * 365 / dias_anio_computables(df)


def media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    """Media ponderada con el factor de elevación de la encuesta."""
    return float(np.average(valores, weights=pesos))


def cargar_ees_unificada() -> pd.DataFrame:
    """Lee 2018 y 2022, calcula el salario anual y los apila en una sola tabla."""
    partes = []
    for anio in ANIOS:
        df = leer_ees(anio)
        df["salario_anual"] = calcular_salario_anual(df)
        partes.append(df)

    columnas_comunes = sorted(set(partes[0].columns) & set(partes[1].columns))
    return pd.concat([p[columnas_comunes] for p in partes], ignore_index=True)
