"""Funciones estadísticas ponderadas con el factor de elevación de la encuesta (FACTOTAL).

En una encuesta por muestreo cada registro representa a un número distinto de trabajadores,
así que las medias, medianas y percentiles deben ponderarse para describir a la población.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cuantil_ponderado(valores, pesos, q):
    """Cuantil q (entre 0 y 1, o lista de ellos) ponderado por `pesos`."""
    valores = np.asarray(valores, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    orden = np.argsort(valores)
    valores, pesos = valores[orden], pesos[orden]
    acumulado = np.cumsum(pesos) / pesos.sum()
    return np.interp(q, acumulado, valores)


def mediana_ponderada(valores, pesos):
    return float(cuantil_ponderado(valores, pesos, 0.5))


def resumen_ponderado(df, por, valor="salario_anual", peso="FACTOTAL", min_registros=30):
    """Tabla con registros, % de población, media, P25, mediana y P75 ponderados por grupo.

    Se descartan los grupos con menos de `min_registros` registros (estimación poco fiable).
    """
    filas = []
    total_peso = df[peso].sum()
    for grupo, sub in df.groupby(por, observed=True):
        if len(sub) < min_registros:
            continue
        p25, p50, p75 = cuantil_ponderado(sub[valor], sub[peso], [0.25, 0.5, 0.75])
        filas.append({
            **(dict(zip(por, grupo)) if isinstance(por, list) else {por: grupo}),
            "registros": len(sub),
            "poblacion_%": 100 * sub[peso].sum() / total_peso,
            "media": np.average(sub[valor], weights=sub[peso]),
            "p25": p25,
            "mediana": p50,
            "p75": p75,
        })
    return pd.DataFrame(filas)


def v_cramer(x, y):
    """V de Cramér: asociación entre dos variables categóricas (0 = ninguna, 1 = total)."""
    tabla = pd.crosstab(x, y)
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def eta_cuadrado(categoria, valor, pesos=None):
    """Proporción de la varianza de `valor` explicada por los grupos de `categoria` (0 a 1)."""
    df = pd.DataFrame({"c": categoria, "v": valor, "w": 1.0 if pesos is None else pesos})
    media_total = np.average(df["v"], weights=df["w"])
    grupos = df.groupby("c", observed=True).apply(
        lambda g: pd.Series({"w": g["w"].sum(), "m": np.average(g["v"], weights=g["w"])}),
        include_groups=False,
    )
    entre = (grupos["w"] * (grupos["m"] - media_total) ** 2).sum()
    total = (df["w"] * (df["v"] - media_total) ** 2).sum()
    return float(entre / total)
