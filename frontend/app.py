"""
Interfaz Streamlit del sistema de prediccion de abandono estudiantil
para la Universidad Autonoma Metropolitana, Unidad Iztapalapa.

La interfaz consulta al backend FastAPI (por defecto en
http://localhost:8000) a traves de HTTP. La URL puede sobreescribirse
con la variable de entorno API_URL.
"""

import io
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="UAM-I | Riesgo de abandono",
    page_icon=None,
    layout="wide",
)

# ----------------------------- helpers ---------------------------------

def _color_riesgo(nivel: str) -> str:
    return {"Bajo": "#16a34a", "Medio": "#f59e0b", "Alto": "#dc2626"}.get(nivel, "#6b7280")


def check_api() -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None


def predict_single(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def predict_batch(alumnos: list[dict]) -> dict:
    r = requests.post(
        f"{API_URL}/predict",
        json={"alumnos": alumnos},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def get_model_info() -> dict:
    r = requests.get(f"{API_URL}/model/info", timeout=5)
    r.raise_for_status()
    return r.json()


# -------------------------- encabezado ---------------------------------
st.title("Prediccion de Riesgo de Abandono - UAM Iztapalapa")
st.caption(
    "Herramienta para estimar la probabilidad de abandono escolar de los "
    "alumnos de la UAM-I a partir de su historial academico."
)

with st.sidebar:
    st.header("Estado del servicio")
    health = check_api()
    if health and health.get("modelo_cargado"):
        st.success(f"API conectada - modelo v{health.get('version', '?')}")
    elif health:
        st.warning("API disponible pero sin modelo cargado.")
        st.info("Ejecuta: python model/train_model.py")
    else:
        st.error(f"No se pudo conectar al backend en {API_URL}")
    st.markdown("---")
    st.markdown(f"**API URL:** `{API_URL}`")
    st.markdown(
        "Para usar otro host/puerto, define la variable de entorno `API_URL` "
        "antes de lanzar Streamlit."
    )
    st.markdown("---")
    st.caption("Universidad Autonoma Metropolitana, Unidad Iztapalapa")

# --------------------------- pestanas -----------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Alumno individual", "Carga por lotes (CSV)", "Informacion del modelo"]
)

# ============================ TAB 1 ======================================
with tab1:
    st.subheader("Historial academico del alumno")
    with st.form("form_alumno"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", 15, 80, 21)
            trimestre = st.number_input("Trimestre cursado", 1, 12, 4)
            promedio = st.slider("Promedio general (0 - 10)", 0.0, 10.0, 7.8, 0.1)
            asistencias = st.slider("% de asistencias", 0.0, 100.0, 80.0, 1.0)
        with c2:
            no_acreditadas = st.number_input("Materias no acreditadas (NA)", 0, 20, 1)
            creditos = st.number_input("Creditos acumulados", 0, 500, 150)
            horas_estudio = st.slider("Horas de estudio / semana", 0.0, 80.0, 10.0, 0.5)
            beca_sel = st.selectbox(
                "Cuenta con beca?",
                options=[("No", 0), ("Si", 1)],
                format_func=lambda x: x[0],
            )
        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        payload = {
            "edad": int(edad),
            "trimestre": int(trimestre),
            "promedio_general": float(promedio),
            "asistencias_porcentaje": float(asistencias),
            "materias_no_acreditadas": int(no_acreditadas),
            "creditos_acumulados": int(creditos),
            "horas_estudio_semana": float(horas_estudio),
            "beca": int(beca_sel[1]),
        }
        with st.spinner("Consultando modelo..."):
            try:
                resultado = predict_single(payload)
            except requests.HTTPError as e:
                st.error(f"Error del backend: {e.response.text}")
                resultado = None
            except requests.RequestException as e:
                st.error(f"No se pudo contactar al backend: {e}")
                resultado = None

        if resultado:
            proba = resultado["probabilidad_abandono"]
            nivel = resultado["nivel_riesgo"]
            color = _color_riesgo(nivel)

            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilidad de abandono", f"{proba * 100:.1f}%")
            m2.metric(
                "Prediccion",
                "ABANDONA" if resultado["prediccion"] == 1 else "CONTINUA",
            )
            m3.markdown(
                f"<div style='padding:1em;border-radius:8px;background:{color};"
                f"color:white;text-align:center;font-weight:bold'>"
                f"Riesgo: {nivel}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Interpretacion")
            st.info(resultado["explicacion"])

            st.markdown("### Variables determinantes")
            df_fact = pd.DataFrame(resultado["factores_principales"])
            if not df_fact.empty:
                fig = px.bar(
                    df_fact,
                    x="impacto",
                    y="label",
                    orientation="h",
                    color="direccion",
                    color_discrete_map={"aumenta": "#dc2626", "reduce": "#16a34a"},
                    title="Contribucion de cada variable a la prediccion",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_fact, use_container_width=True)

# ============================ TAB 2 ======================================
with tab2:
    st.subheader("Prediccion por lotes desde archivo CSV")
    st.markdown(
        "El archivo debe contener las columnas: "
        "`edad, trimestre, promedio_general, asistencias_porcentaje, "
        "materias_no_acreditadas, creditos_acumulados, horas_estudio_semana, beca`."
    )
    st.caption("Hay un archivo de prueba en `data/muestra_alumnos.csv`.")

    archivo = st.file_uploader("Sube el archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            df = None

        if df is not None:
            st.markdown(f"**Registros detectados:** {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Ejecutar prediccion del lote", type="primary"):
                with st.spinner(f"Procesando {len(df)} alumnos..."):
                    try:
                        respuesta = predict_batch(df.to_dict(orient="records"))
                    except requests.HTTPError as e:
                        st.error(f"Error del backend: {e.response.text}")
                        respuesta = None
                    except requests.RequestException as e:
                        st.error(f"No se pudo contactar al backend: {e}")
                        respuesta = None

                if respuesta:
                    total = respuesta["total"]
                    alto = respuesta["en_alto_riesgo"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", total)
                    c2.metric("En riesgo alto", alto)
                    c3.metric(
                        "% en riesgo alto",
                        f"{(alto / total) * 100:.1f}%" if total else "0%",
                    )

                    resultados = pd.DataFrame(
                        [
                            {
                                "probabilidad_abandono": r["probabilidad_abandono"],
                                "prediccion": r["prediccion"],
                                "nivel_riesgo": r["nivel_riesgo"],
                            }
                            for r in respuesta["resultados"]
                        ]
                    )
                    df_out = pd.concat([df.reset_index(drop=True), resultados], axis=1)

                    st.markdown("### Resultados")
                    st.dataframe(df_out, use_container_width=True)

                    fig = px.histogram(
                        df_out,
                        x="probabilidad_abandono",
                        color="nivel_riesgo",
                        nbins=20,
                        color_discrete_map={
                            "Bajo": "#16a34a",
                            "Medio": "#f59e0b",
                            "Alto": "#dc2626",
                        },
                        title="Distribucion de probabilidad de abandono",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    buf = io.StringIO()
                    df_out.to_csv(buf, index=False)
                    st.download_button(
                        "Descargar resultados en CSV",
                        data=buf.getvalue(),
                        file_name="predicciones_uami.csv",
                        mime="text/csv",
                    )

# ============================ TAB 3 ======================================
with tab3:
    st.subheader("Informacion del modelo")
    try:
        info = get_model_info()
    except requests.RequestException as e:
        st.error(f"No se pudo obtener la informacion del modelo: {e}")
        info = None

    if info:
        c1, c2 = st.columns(2)
        c1.markdown(f"**Institucion:** {info.get('institucion', '-')}")
        c1.markdown(f"**Algoritmo:** {info['algoritmo']}")
        c1.markdown(f"**Version:** {info['version']}")
        c1.markdown(f"**Variable objetivo:** {info['target']}")
        metricas = info.get("metricas", {})
        c2.metric("Accuracy", metricas.get("accuracy", "?"))
        c2.metric("ROC-AUC", metricas.get("roc_auc", "?"))

        st.markdown("### Importancia de las variables")
        importancias = info.get("feature_importances", {})
        if importancias:
            df_imp = (
                pd.DataFrame(
                    {
                        "variable": list(importancias.keys()),
                        "importancia": list(importancias.values()),
                    }
                )
                .sort_values("importancia", ascending=True)
            )
            fig = px.bar(
                df_imp,
                x="importancia",
                y="variable",
                orientation="h",
                title="Importancia global de cada variable",
                color="importancia",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Reporte de clasificacion (conjunto de prueba)")
        st.json(metricas.get("classification_report", {}))
