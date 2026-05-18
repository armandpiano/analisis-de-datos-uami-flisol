"""
Entrenamiento del modelo de prediccion de abandono estudiantil
para la Universidad Autonoma Metropolitana, Unidad Iztapalapa (UAM-I).

Variables (8) basadas en el historial academico del alumno:
    edad                       anios
    trimestre                  trimestre cursado (1 a 12)
    promedio_general           escala 0 - 10
    asistencias_porcentaje     0 - 100
    materias_no_acreditadas    UEAS marcadas NA
    creditos_acumulados        creditos del plan de estudios
    horas_estudio_semana       horas dedicadas fuera de clase
    beca                       0 = no, 1 = si

Objetivo: abandono (0 = continua, 1 = abandona).

Nota: el dataset es sintetico. No se utilizan datos reales del SAE ni del
sistema de servicios escolares de la UAM-I. Sirve unicamente como material
didactico para el taller.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

SEMILLA = 42
np.random.seed(SEMILLA)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Orden canonico de las features. El modelo espera estas columnas en
# este orden al hacer predicciones.
FEATURES = [
    "edad",
    "trimestre",
    "promedio_general",
    "asistencias_porcentaje",
    "materias_no_acreditadas",
    "creditos_acumulados",
    "horas_estudio_semana",
    "beca",
]


def generar_datos_sinteticos(n: int = 2000) -> pd.DataFrame:
    """Devuelve un DataFrame con n alumnos ficticios para entrenar."""
    edad = np.random.normal(loc=22, scale=3, size=n).clip(17, 50).astype(int)
    trimestre = np.random.randint(1, 13, size=n)  # UAM-I: 1 a 12 trimestres
    promedio = np.random.normal(loc=7.8, scale=1.2, size=n).clip(0, 10).round(2)
    asistencias = np.random.normal(loc=78, scale=15, size=n).clip(0, 100).round(1)
    no_acreditadas = np.random.poisson(lam=1.2, size=n).clip(0, 10)
    creditos = np.random.normal(loc=200, scale=120, size=n).clip(0, 500).astype(int)
    horas_estudio = np.random.normal(loc=10, scale=5, size=n).clip(0, 40).round(1)
    beca = np.random.binomial(1, 0.30, size=n)  # ~30% con beca

    df = pd.DataFrame(
        {
            "edad": edad,
            "trimestre": trimestre,
            "promedio_general": promedio,
            "asistencias_porcentaje": asistencias,
            "materias_no_acreditadas": no_acreditadas,
            "creditos_acumulados": creditos,
            "horas_estudio_semana": horas_estudio,
            "beca": beca,
        }
    )

    # Score de riesgo heuristico (no normalizado). Promedio bajo, baja
    # asistencia y UEAS no acreditadas suben el riesgo; horas de estudio
    # y beca lo reducen.
    score = (
        (10 - df["promedio_general"]) * 0.75
        + (100 - df["asistencias_porcentaje"]) * 0.04
        + df["materias_no_acreditadas"] * 0.80
        - df["horas_estudio_semana"] * 0.08
        - df["beca"] * 0.60
        + np.random.normal(0, 1.0, size=n)
    )

    # Calibrar el umbral para obtener ~22% de positivos (cercano a la
    # tasa de abandono reportada en universidades publicas mexicanas).
    umbral = np.quantile(score, 0.78)
    df["abandono"] = (score >= umbral).astype(int)
    return df


def entrenar_modelo(df: pd.DataFrame):
    """Entrena un Random Forest y devuelve (modelo, metricas)."""
    X = df[FEATURES]
    y = df["abandono"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEMILLA,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "porcentaje_positivos": round(float(y.mean()) * 100, 2),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "feature_importances": dict(
            zip(FEATURES, [round(float(v), 4) for v in modelo.feature_importances_])
        ),
    }
    return modelo, metricas


def guardar_artefactos(modelo, metricas: dict, df: pd.DataFrame) -> None:
    """Persiste el modelo entrenado, su metadata y los datasets."""
    model_path = MODEL_DIR / "model.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"
    data_path = DATA_DIR / "alumnos_sinteticos.csv"
    sample_path = DATA_DIR / "muestra_alumnos.csv"

    joblib.dump(modelo, model_path)

    metadata = {
        "institucion": "Universidad Autonoma Metropolitana - Unidad Iztapalapa",
        "algoritmo": "RandomForestClassifier",
        "version": "1.0.0",
        "features": FEATURES,
        "target": "abandono",
        "n_estimators": modelo.n_estimators,
        "max_depth": modelo.max_depth,
        "metricas": metricas,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    df.to_csv(data_path, index=False)
    df.drop(columns=["abandono"]).sample(20, random_state=SEMILLA).to_csv(
        sample_path, index=False
    )

    print(f"[OK] Modelo guardado en       : {model_path}")
    print(f"[OK] Metadata guardada en     : {meta_path}")
    print(f"[OK] Dataset sintetico        : {data_path}")
    print(f"[OK] Muestra para carga lote  : {sample_path}")


def main() -> None:
    print(">> Generando datos sinteticos...")
    df = generar_datos_sinteticos(n=2000)
    print(f"   {len(df)} alumnos. Tasa de abandono simulada: {df['abandono'].mean():.2%}")

    print(">> Entrenando Random Forest...")
    modelo, metricas = entrenar_modelo(df)
    print(f"   Accuracy: {metricas['accuracy']} | ROC-AUC: {metricas['roc_auc']}")

    print(">> Persistiendo artefactos...")
    guardar_artefactos(modelo, metricas, df)
    print(">> Entrenamiento concluido.")


if __name__ == "__main__":
    main()
