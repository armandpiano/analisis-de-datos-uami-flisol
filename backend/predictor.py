"""
Capa de servicio que envuelve al modelo entrenado.

Responsabilidades:
- Cargar el modelo y la metadata desde disco al iniciar la API.
- Convertir entradas a DataFrames con el orden de columnas que el modelo espera.
- Calcular probabilidad de abandono, nivel de riesgo y factores principales (SHAP).
- Generar una explicacion textual breve.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

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

# Etiquetas para mostrar al usuario final.
FEATURE_LABELS = {
    "edad": "Edad",
    "trimestre": "Trimestre",
    "promedio_general": "Promedio general",
    "asistencias_porcentaje": "% de asistencias",
    "materias_no_acreditadas": "Materias NA",
    "creditos_acumulados": "Creditos acumulados",
    "horas_estudio_semana": "Horas de estudio / semana",
    "beca": "Cuenta con beca",
}


class PredictorService:
    """Wrapper alrededor del modelo. Se instancia una vez al iniciar la API."""

    def __init__(self) -> None:
        self.model = None
        self.metadata: Dict = {}
        self._shap_explainer = None

    def cargar(self) -> None:
        """Carga el modelo y la metadata desde disco."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {MODEL_PATH}. "
                "Ejecuta primero: python model/train_model.py"
            )
        self.model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    def esta_cargado(self) -> bool:
        return self.model is not None

    def version(self) -> Optional[str]:
        return self.metadata.get("version")

    def _a_dataframe(self, registros: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(registros)
        faltantes = [f for f in FEATURES if f not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {faltantes}")
        return df[FEATURES].copy()

    def predecir(self, registros: List[Dict]) -> List[Dict]:
        """Devuelve, por cada alumno: probabilidad, prediccion, riesgo y factores."""
        if not self.esta_cargado():
            raise RuntimeError("El modelo no esta cargado.")

        df = self._a_dataframe(registros)
        proba = self.model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # SHAP solo para volumenes pequenos. Para batches grandes se aproxima
        # con la importancia global de cada variable.
        usar_shap = len(df) <= 50
        shap_values = self._calcular_shap(df) if usar_shap else None

        resultados: List[Dict] = []
        for i in range(len(df)):
            factores = self._factores_principales(
                df.iloc[i], shap_values[i] if shap_values is not None else None
            )
            resultados.append(
                {
                    "probabilidad_abandono": round(float(proba[i]), 4),
                    "prediccion": int(pred[i]),
                    "nivel_riesgo": self._nivel_riesgo(proba[i]),
                    "factores_principales": factores,
                    "explicacion": self._explicacion_textual(
                        df.iloc[i], proba[i], factores
                    ),
                }
            )
        return resultados

    @staticmethod
    def _nivel_riesgo(p: float) -> str:
        if p < 0.30:
            return "Bajo"
        if p < 0.60:
            return "Medio"
        return "Alto"

    def _calcular_shap(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Calcula valores SHAP para la clase positiva."""
        try:
            import shap

            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self.model)
            sv = self._shap_explainer.shap_values(df)
            if isinstance(sv, list):
                return np.array(sv[1])
            arr = np.array(sv)
            if arr.ndim == 3:
                return arr[:, :, 1]
            return arr
        except Exception:
            return None

    def _factores_principales(
        self, fila: pd.Series, shap_row: Optional[np.ndarray]
    ) -> List[Dict]:
        """Top 3 variables que mas pesan en esta prediccion."""
        if shap_row is not None:
            contribuciones = list(zip(FEATURES, shap_row))
        else:
            importancias = self.model.feature_importances_
            contribuciones = list(zip(FEATURES, importancias * fila.values))

        contribuciones.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contribuciones[:3]
        return [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "valor": float(fila[name]),
                "impacto": round(float(val), 4),
                "direccion": "aumenta" if val > 0 else "reduce",
            }
            for name, val in top
        ]

    @staticmethod
    def _explicacion_textual(
        fila: pd.Series, proba: float, factores: List[Dict]
    ) -> str:
        riesgo = PredictorService._nivel_riesgo(proba)
        porc = f"{proba * 100:.1f}%"
        partes = [
            f"El alumno presenta una probabilidad de abandono de {porc} "
            f"(riesgo {riesgo.lower()})."
        ]
        if factores:
            descripciones = []
            for f in factores:
                etiqueta = f["label"]
                valor = f["valor"]
                accion = "aumenta" if f["direccion"] == "aumenta" else "reduce"
                descripciones.append(f"{etiqueta} = {valor:g} ({accion} el riesgo)")
            partes.append("Variables determinantes: " + "; ".join(descripciones) + ".")
        return " ".join(partes)


predictor_service = PredictorService()
