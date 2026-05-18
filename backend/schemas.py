"""
Modelos Pydantic usados por la API. Definen el contrato JSON
de entrada y salida de los endpoints.

Las variables siguen la nomenclatura academica de la UAM-I:
- trimestre (la UAM opera por trimestres)
- promedio_general en escala 0 - 10
- materias_no_acreditadas (UEAS marcadas NA)
- creditos_acumulados
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Alumno(BaseModel):
    """Datos academicos de un alumno UAM-I."""

    edad: int = Field(..., ge=15, le=80, description="Edad en anios")
    trimestre: int = Field(..., ge=1, le=12, description="Trimestre cursado (1 a 12)")
    promedio_general: float = Field(
        ..., ge=0.0, le=10.0, description="Promedio general (escala 0 a 10)"
    )
    asistencias_porcentaje: float = Field(
        ..., ge=0.0, le=100.0, description="Porcentaje de asistencias"
    )
    materias_no_acreditadas: int = Field(
        ..., ge=0, le=20, description="UEAS marcadas como NA en el historial"
    )
    creditos_acumulados: int = Field(
        ..., ge=0, le=500, description="Creditos acumulados del plan de estudios"
    )
    horas_estudio_semana: float = Field(
        ..., ge=0.0, le=80.0, description="Horas de estudio independientes por semana"
    )
    beca: int = Field(..., ge=0, le=1, description="1 si cuenta con beca, 0 si no")


class PrediccionRespuesta(BaseModel):
    """Resultado de la prediccion para un alumno."""

    probabilidad_abandono: float = Field(..., description="Probabilidad entre 0 y 1")
    prediccion: int = Field(..., description="1 = abandona, 0 = continua")
    nivel_riesgo: str = Field(..., description="Bajo / Medio / Alto")
    factores_principales: List[dict] = Field(
        default_factory=list,
        description="Variables con mayor peso en la prediccion",
    )
    explicacion: str = Field(..., description="Explicacion en texto plano")


class LotePrediccionEntrada(BaseModel):
    """Entrada para el endpoint de prediccion por lotes."""

    alumnos: List[Alumno]


class LotePrediccionRespuesta(BaseModel):
    """Respuesta del endpoint de prediccion por lotes."""

    total: int
    en_alto_riesgo: int
    resultados: List[PrediccionRespuesta]


class ModelInfo(BaseModel):
    """Metadata del modelo expuesta por /model/info."""

    institucion: str
    algoritmo: str
    version: str
    features: List[str]
    target: str
    metricas: dict
    feature_importances: dict


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""

    status: str
    modelo_cargado: bool
    version: Optional[str] = None
