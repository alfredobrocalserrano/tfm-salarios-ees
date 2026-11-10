"""Preparación de los datos para la modelización.

Este módulo lo usan el notebook 03 (para generar los conjuntos de entrenamiento y prueba)
y más adelante la API, para transformar el perfil que introduce el usuario exactamente igual
que los datos de entrenamiento.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.datos import DATA_PROCESSED, cargar_dataset_procesado
from src.estadistica import cuantil_ponderado

# IPC general, base 2021, media anual (INE). Sirve para expresar todos los salarios en euros de 2022.
IPC_MEDIA_ANUAL = {2018: 96.6, 2019: 97.3, 2020: 97.0, 2021: 100.0, 2022: 108.4}
ANIO_REFERENCIA = 2022

# Variables de entrada del modelo
VARIABLES_CATEGORICAS = [
    "ocupacion", "sector", "estudios", "region", "edad", "sexo", "nacionalidad",
    "jornada", "contrato", "tamano_empresa", "control", "responsabilidad", "convenio", "mercado",
]
VARIABLES_NUMERICAS = ["horas_semanales", "antiguedad"]
VARIABLES_MODELO = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS

# Variables que un usuario puede responder con facilidad en la web
# (se excluyen el tipo de convenio y el mercado principal de la empresa).
VARIABLES_APP = [v for v in VARIABLES_MODELO if v not in ("convenio", "mercado")]

OBJETIVO = "log_salario"
PESO = "FACTOTAL"
GRUPO = "IDENCCC"  # centro de cotización: los trabajadores de una misma empresa van juntos

# Códigos de tamaño que el INE agrupa por anonimización y no corresponden a un tamaño concreto
TAMANOS_SIN_DESAGREGAR = ["Sin desagregar (todos los tamaños)", "50 o más (sin desagregar)"]

PERCENTIL_RECORTE = 0.999

CARPETA_MODELO = DATA_PROCESSED / "modelo"


def deflactar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade `salario_real`: el salario anual expresado en euros de 2022 según el IPC."""
    df = df.copy()
    factor = df["anio"].map(lambda a: IPC_MEDIA_ANUAL[ANIO_REFERENCIA] / IPC_MEDIA_ANUAL[a])
    df["salario_real"] = df["salario_anual"] * factor
    return df


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables numéricas derivadas: horas semanales pactadas y años de antigüedad."""
    df = df.copy()
    df["horas_semanales"] = df["JSP1"] + df["JSP2"] / 60
    df["antiguedad"] = df["ANOANTI"] + df["MESANTI"] / 12
    return df


def recortar_extremos(df: pd.DataFrame, percentil: float = PERCENTIL_RECORTE):
    """Limita (no elimina) los salarios por encima del percentil indicado, calculado por año.

    Devuelve el DataFrame y un diccionario con el umbral aplicado cada año.
    """
    df = df.copy()
    umbrales = {}
    for anio, g in df.groupby("anio"):
        umbral = float(cuantil_ponderado(g["salario_real"], g[PESO], percentil))
        umbrales[int(anio)] = umbral
        mascara = (df["anio"] == anio) & (df["salario_real"] > umbral)
        df.loc[mascara, "salario_real"] = umbral
    return df, umbrales


def preparar(df: pd.DataFrame):
    """Aplica todos los pasos y devuelve el dataset de modelado y un registro de lo hecho."""
    registro = {"registros_iniciales": int(len(df))}

    df = deflactar(construir_variables(df))

    sin_desagregar = df["tamano_empresa"].isin(TAMANOS_SIN_DESAGREGAR)
    registro["excluidos_tamano_sin_desagregar"] = int(sin_desagregar.sum())
    df = df[~sin_desagregar].copy()
    df["tamano_empresa"] = df["tamano_empresa"].cat.remove_categories(TAMANOS_SIN_DESAGREGAR)

    registro["recortados_por_arriba"] = {
        int(a): int((g["salario_real"] > cuantil_ponderado(g["salario_real"], g[PESO], PERCENTIL_RECORTE)).sum())
        for a, g in df.groupby("anio")
    }
    df, umbrales = recortar_extremos(df)
    registro["umbral_recorte"] = {a: round(u, 2) for a, u in umbrales.items()}

    df[OBJETIVO] = np.log(df["salario_real"])
    registro["registros_finales"] = int(len(df))

    columnas = ["anio", GRUPO, PESO, *VARIABLES_MODELO, "salario_real", OBJETIVO]
    return df[columnas].reset_index(drop=True), registro


def dividir_por_empresa(df: pd.DataFrame, proporcion_test: float = 0.2, semilla: int = 42):
    """Separa entrenamiento y prueba de modo que una empresa no aparezca en los dos conjuntos."""
    divisor = GroupShuffleSplit(n_splits=1, test_size=proporcion_test, random_state=semilla)
    idx_train, idx_test = next(divisor.split(df, groups=df[GRUPO]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


def generar_conjuntos(guardar: bool = True):
    """Genera train y test (2022) y el conjunto de 2018 para la validación temporal."""
    datos, registro = preparar(cargar_dataset_procesado())
    train, test = dividir_por_empresa(datos[datos["anio"] == 2022])
    conjuntos = {"train_2022": train, "test_2022": test, "ees_2018": datos[datos["anio"] == 2018].reset_index(drop=True)}
    registro["tamanos"] = {k: int(len(v)) for k, v in conjuntos.items()}

    if guardar:
        CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
        for nombre, conjunto in conjuntos.items():
            conjunto.to_parquet(CARPETA_MODELO / f"{nombre}.parquet", index=False)
        metadatos = {
            **registro,
            "ipc_media_anual": IPC_MEDIA_ANUAL,
            "variables_categoricas": VARIABLES_CATEGORICAS,
            "variables_numericas": VARIABLES_NUMERICAS,
            "variables_app": VARIABLES_APP,
            "objetivo": OBJETIVO,
            "peso": PESO,
        }
        (CARPETA_MODELO / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))
    return conjuntos, registro


def cargar_conjuntos():
    """Carga train, test y 2018 desde disco, o los genera si no existen (p. ej. en Colab)."""
    nombres = ["train_2022", "test_2022", "ees_2018"]
    if all((CARPETA_MODELO / f"{n}.parquet").exists() for n in nombres):
        return {n: pd.read_parquet(CARPETA_MODELO / f"{n}.parquet") for n in nombres}
    conjuntos, _ = generar_conjuntos(guardar=True)
    return conjuntos
