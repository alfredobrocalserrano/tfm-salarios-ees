"""Modelos, métricas y el modelo final de rango salarial.

Lo usan el notebook 04 (comparación de modelos), el 05 (interpretabilidad) y la API.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer

from src.datos import RAIZ
from src.estadistica import cuantil_ponderado
from src.preparacion import (
    OBJETIVO, PESO, VARIABLES_APP, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS, dividir_por_empresa,
)

CARPETA_MODELOS = RAIZ / "models"
SEMILLA = 42

# Hiperparámetros de LightGBM encontrados con Optuna en el notebook 04
PARAMETROS_LGBM = {
    "learning_rate": 0.0208,
    "num_leaves": 213,
    "min_child_samples": 15,
    "colsample_bytree": 0.629,
    "subsample": 0.590,
    "subsample_freq": 1,
    "reg_lambda": 0.006,
    "cat_smooth": 30.9,
    "n_estimators": 494,
}


# ---------------------------------------------------------------------------
# Métricas (todas ponderadas con FACTOTAL)
# ---------------------------------------------------------------------------

def metricas(y_log, pred_log, pesos) -> dict:
    """Métricas de calidad de una predicción del logaritmo del salario.

    - R² (escala logarítmica): proporción de la variabilidad del salario explicada.
    - MAE (€): error absoluto medio en euros.
    - Error mediano (%): la mitad de las predicciones se equivoca menos que este porcentaje.
    - % a menos del 20 %: proporción de predicciones con un error inferior al 20 %.
    """
    y_log, pred_log, pesos = np.asarray(y_log), np.asarray(pred_log), np.asarray(pesos)
    media = np.average(y_log, weights=pesos)
    r2 = 1 - np.average((y_log - pred_log) ** 2, weights=pesos) / np.average((y_log - media) ** 2, weights=pesos)
    real, prediccion = np.exp(y_log), np.exp(pred_log)
    error_rel = np.abs(prediccion - real) / real
    return {
        "R²": r2,
        "MAE (€)": np.average(np.abs(prediccion - real), weights=pesos),
        "Error mediano (%)": 100 * float(cuantil_ponderado(error_rel, pesos, 0.5)),
        "% a menos del 20 %": 100 * np.average(error_rel <= 0.20, weights=pesos),
    }


def cobertura(y_log, inferior_log, superior_log, pesos) -> float:
    """% de casos (ponderado) cuyo salario real cae dentro del intervalo."""
    dentro = (np.asarray(y_log) >= inferior_log) & (np.asarray(y_log) <= superior_log)
    return 100 * float(np.average(dentro, weights=pesos))


# ---------------------------------------------------------------------------
# Modelos que se comparan
# ---------------------------------------------------------------------------

def crear_ridge(variables_cat=VARIABLES_CATEGORICAS, variables_num=VARIABLES_NUMERICAS, alpha=1.0):
    """Regresión lineal regularizada: categorías en one-hot y curvas suaves (splines) para las numéricas."""
    preprocesado = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_cat),
        ("numericas", SplineTransformer(n_knots=5, degree=3), variables_num),
    ])
    return make_pipeline(preprocesado, Ridge(alpha=alpha))


def crear_random_forest(variables_cat=VARIABLES_CATEGORICAS):
    """Random Forest: las categorías se codifican como números enteros."""
    preprocesado = ColumnTransformer(
        [("categoricas", OrdinalEncoder(), variables_cat)], remainder="passthrough"
    )
    bosque = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEMILLA
    )
    return make_pipeline(preprocesado, bosque)


def crear_lgbm(**parametros):
    """LightGBM con tratamiento nativo de las variables categóricas de pandas."""
    base = {"verbose": -1, "random_state": SEMILLA}
    return lgb.LGBMRegressor(**{**base, **parametros})


# ---------------------------------------------------------------------------
# Modelo final: rango salarial (P10, P50, P90) con calibración conformal
# ---------------------------------------------------------------------------

class ModeloRangoSalarial:
    """Tres modelos LightGBM cuantílicos (P10, P50, P90) calibrados con predicción conformal.

    El intervalo P10–P90 se amplía lo justo (`ajuste`) para que contenga el salario real
    del 80 % de los trabajadores de un conjunto de calibración que el modelo no ha visto
    (Conformalized Quantile Regression, Romano et al., 2019).
    """

    CUANTILES = (0.10, 0.50, 0.90)
    COBERTURA_OBJETIVO = 0.80

    def __init__(self, variables=VARIABLES_APP, parametros=None):
        self.variables = list(variables)
        self.parametros = dict(parametros or PARAMETROS_LGBM)
        self.modelos = {}
        self.ajuste = 0.0
        self.categorias = {}

    def entrenar(self, train: pd.DataFrame, proporcion_calibracion: float = 0.2):
        """Entrena con una parte del conjunto y calibra el intervalo con el resto (separado por empresa)."""
        ajuste_df, calibracion = dividir_por_empresa(train, proporcion_calibracion, semilla=7)
        self.categorias = {
            v: list(train[v].cat.categories) for v in self.variables if isinstance(train[v].dtype, pd.CategoricalDtype)
        }
        for q in self.CUANTILES:
            modelo = crear_lgbm(objective="quantile", alpha=q, **self.parametros)
            modelo.fit(ajuste_df[self.variables], ajuste_df[OBJETIVO], sample_weight=ajuste_df[PESO])
            self.modelos[q] = modelo

        pred = self._predecir_log(calibracion)
        y = calibracion[OBJETIVO].to_numpy()
        puntuaciones = np.maximum(pred[0.10] - y, y - pred[0.90])
        self.ajuste = float(cuantil_ponderado(puntuaciones, calibracion[PESO], self.COBERTURA_OBJETIVO))
        return self

    def _predecir_log(self, df: pd.DataFrame) -> dict:
        return {q: m.predict(df[self.variables]) for q, m in self.modelos.items()}

    def predecir_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción en escala logarítmica, con el intervalo ya calibrado."""
        pred = self._predecir_log(df)
        p10 = pred[0.10] - self.ajuste
        p90 = pred[0.90] + self.ajuste
        p50 = np.clip(pred[0.50], p10, p90)  # garantiza P10 <= P50 <= P90
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=df.index)

    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rango salarial en euros de 2022."""
        return np.exp(self.predecir_log(df))

    # --- guardado y carga -------------------------------------------------
    def guardar(self, carpeta: Path = CARPETA_MODELOS):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        for q, modelo in self.modelos.items():
            modelo.booster_.save_model(str(carpeta / f"lgbm_p{int(q * 100)}.txt"))
        (carpeta / "modelo_rango.json").write_text(json.dumps({
            "variables": self.variables,
            "parametros": self.parametros,
            "ajuste_conformal": self.ajuste,
            "cobertura_objetivo": self.COBERTURA_OBJETIVO,
            "categorias": self.categorias,
        }, indent=2, ensure_ascii=False))
        return carpeta

    @classmethod
    def cargar(cls, carpeta: Path = CARPETA_MODELOS):
        carpeta = Path(carpeta)
        info = json.loads((carpeta / "modelo_rango.json").read_text())
        modelo = cls(info["variables"], info["parametros"])
        modelo.ajuste = info["ajuste_conformal"]
        modelo.categorias = info["categorias"]
        modelo.modelos = {q: _BoosterComoModelo(lgb.Booster(model_file=str(carpeta / f"lgbm_p{int(q * 100)}.txt")))
                          for q in cls.CUANTILES}
        return modelo


class _BoosterComoModelo:
    """Envoltorio mínimo para que un Booster cargado de disco tenga `.predict()` como el modelo original."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(X)


def cargar_o_entrenar_modelo(conjuntos=None, carpeta: Path = CARPETA_MODELOS) -> ModeloRangoSalarial:
    """Carga el modelo final si ya está guardado; si no (p. ej. en Colab), lo entrena y lo guarda."""
    if (Path(carpeta) / "modelo_rango.json").exists():
        return ModeloRangoSalarial.cargar(carpeta)
    if conjuntos is None:
        from src.preparacion import cargar_conjuntos
        conjuntos = cargar_conjuntos()
    modelo = ModeloRangoSalarial().entrenar(conjuntos["train_2022"])
    modelo.guardar(carpeta)
    return modelo
