"""
API del sistema de prediccion de abandono estudiantil para la
Universidad Autonoma Metropolitana, Unidad Iztapalapa (UAM-I).

Endpoints:
    GET  /health      healthcheck
    GET  /model/info  metadata del modelo
    POST /predict     prediccion individual o por lotes

Levantar con:
    uvicorn backend.main:app --reload --port 8000
"""

from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import predictor_service
from backend.schemas import (
    Alumno,
    HealthResponse,
    LotePrediccionEntrada,
    LotePrediccionRespuesta,
    ModelInfo,
    PrediccionRespuesta,
)

app = FastAPI(
    title="Prediccion de Abandono Estudiantil - UAM Iztapalapa",
    description=(
        "Servicio para estimar el riesgo de abandono escolar de un alumno "
        "de la UAM-I a partir de su historial academico."
    ),
    version="1.0.0",
)

# CORS abierto para el taller. En produccion se debe restringir el origen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _cargar_modelo() -> None:
    """Carga el modelo al iniciar la API."""
    try:
        predictor_service.cargar()
        print(f"[OK] Modelo cargado. Version: {predictor_service.version()}")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    """Verifica el estado de la API y del modelo."""
    return HealthResponse(
        status="ok",
        modelo_cargado=predictor_service.esta_cargado(),
        version=predictor_service.version(),
    )


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    """Devuelve la metadata del modelo cargado."""
    if not predictor_service.esta_cargado():
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    meta = predictor_service.metadata
    return ModelInfo(
        institucion=meta.get("institucion", "UAM Iztapalapa"),
        algoritmo=meta.get("algoritmo", "desconocido"),
        version=meta.get("version", "0.0.0"),
        features=meta.get("features", []),
        target=meta.get("target", "abandono"),
        metricas=meta.get("metricas", {}),
        feature_importances=meta.get("metricas", {}).get("feature_importances", {}),
    )


@app.post(
    "/predict",
    response_model=Union[PrediccionRespuesta, LotePrediccionRespuesta],
)
def predict(payload: Union[Alumno, LotePrediccionEntrada]):
    """Predice abandono para un alumno o para un lote."""
    if not predictor_service.esta_cargado():
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        if isinstance(payload, Alumno):
            registros = [payload.model_dump()]
            resultado = predictor_service.predecir(registros)[0]
            return PrediccionRespuesta(**resultado)

        registros = [a.model_dump() for a in payload.alumnos]
        if not registros:
            raise HTTPException(status_code=400, detail="La lista de alumnos esta vacia.")
        resultados = predictor_service.predecir(registros)
        en_alto = sum(1 for r in resultados if r["nivel_riesgo"] == "Alto")
        return LotePrediccionRespuesta(
            total=len(resultados),
            en_alto_riesgo=en_alto,
            resultados=[PrediccionRespuesta(**r) for r in resultados],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
