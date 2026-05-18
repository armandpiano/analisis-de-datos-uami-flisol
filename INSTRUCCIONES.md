# INSTRUCCIONES DE USO

Prediccion de Abandono Estudiantil - **UAM Iztapalapa**

Guia paso a paso para **ejecutar** el sistema y entender **como funciona**
por dentro.

---

## PARTE 1: COMO EJECUTARLO

### Requisitos previos

- **Python 3.9 o superior** instalado. Verificar con:
  ```
  python --version
  ```
- Se recomienda **Python 3.11 o 3.12** porque tienen wheels precompilados
  para todas las librerias del proyecto. Python 3.14 funciona pero puede
  requerir mas tiempo de instalacion.
- Conexion a Internet la primera vez (para descargar dependencias).
- Aproximadamente 500 MB libres en disco.

### Opcion A: Ejecucion automatica (recomendada)

#### En Windows

1. Abrir una terminal (PowerShell o CMD) en la carpeta del proyecto:
   ```
   cd C:\ruta\al\student-dropout-predictor
   ```
2. Ejecutar:
   ```
   .\run.bat
   ```
3. Esperar 1 a 2 minutos la primera vez (instalacion y entrenamiento).
4. Se abriran dos ventanas:
   - Una con el backend (FastAPI) en `http://localhost:8000`.
   - Una con el frontend (Streamlit) que abre `http://localhost:8501`
     automaticamente en el navegador.

#### En Linux / macOS

1. Abrir una terminal en la carpeta del proyecto.
2. Otorgar permisos de ejecucion (solo la primera vez):
   ```
   chmod +x run.sh
   ```
3. Ejecutar:
   ```
   ./run.sh
   ```
4. El frontend se abrira en `http://localhost:8501`.

### Opcion B: Ejecucion manual (paso a paso)

```
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo (solo la primera vez)
python model/train_model.py

# 5. Levantar el backend (en una terminal)
uvicorn backend.main:app --reload --port 8000

# 6. Levantar el frontend (en OTRA terminal, con el venv activado)
streamlit run frontend/app.py
```

### Como detener el sistema

- **Frontend (Streamlit)**: `Ctrl + C` en la terminal.
- **Backend (FastAPI)**: cerrar la ventana o `Ctrl + C` en su terminal.

---

## PARTE 2: COMO USAR LA APLICACION

Una vez en `http://localhost:8501` apareceran tres pestanas:

### Pestana 1: Alumno individual

Calcula el riesgo de **un alumno** capturado a mano.

1. Llenar el formulario con los 8 datos academicos (edad, trimestre,
   promedio general, etc.).
2. Click en **Calcular riesgo**.
3. La pagina muestra:
   - **Probabilidad de abandono** en porcentaje.
   - **Prediccion**: ABANDONA o CONTINUA.
   - **Nivel de riesgo**: Bajo (verde) / Medio (amarillo) / Alto (rojo).
   - **Interpretacion** en texto plano.
   - Grafica con las 3 variables que mas pesaron en la prediccion.

**Casos sugeridos para el taller:**

| Caso                  | Promedio | Asistencia | Materias NA | Horas estudio | Beca | Resultado esperado |
|-----------------------|----------|------------|-------------|---------------|------|--------------------|
| Alumno regular        | 9.0      | 95         | 0           | 20            | 1    | Riesgo Bajo        |
| Alumno tipico         | 7.8      | 80         | 1           | 10            | 0    | Riesgo Bajo/Medio  |
| Alumno en riesgo      | 5.5      | 50         | 4           | 3             | 0    | Riesgo Alto        |

### Pestana 2: Carga por lotes (CSV)

Permite predecir **varios alumnos al mismo tiempo** desde un archivo CSV.

1. Click en **Sube el archivo CSV** y seleccionar un archivo.
2. Como archivo de prueba se puede usar: `data/muestra_alumnos.csv`
   (20 alumnos generados aleatoriamente).
3. Click en **Ejecutar prediccion del lote**.
4. La aplicacion muestra:
   - Conteo de alumnos en riesgo alto.
   - Tabla con todas las predicciones.
   - Histograma de la distribucion de probabilidades.
   - Boton para **descargar los resultados en CSV**.

**Formato del archivo CSV** (encabezados obligatorios):
```
edad,trimestre,promedio_general,asistencias_porcentaje,materias_no_acreditadas,creditos_acumulados,horas_estudio_semana,beca
21,4,6.5,70,3,120,5,0
23,7,8.9,90,0,250,15,1
```

### Pestana 3: Informacion del modelo

Muestra los metadatos del modelo entrenado:
- Institucion, algoritmo, version, accuracy y ROC-AUC.
- Grafica con la **importancia global** de cada variable.
- Reporte de clasificacion (precision, recall, F1).

---

## PARTE 3: COMO FUNCIONA POR DENTRO

### Arquitectura general

```
   Navegador
       |
       v
  +-----------+     HTTP/JSON     +-----------+     joblib     +-----------+
  | Streamlit | <---------------> |  FastAPI  | <------------> | model.pkl |
  | (puerto   |                   | (puerto   |                | (Random   |
  |   8501)   |                   |   8000)   |                |  Forest)  |
  +-----------+                   +-----------+                +-----------+
   frontend/app.py                 backend/main.py              model/
```

- El **frontend** (Streamlit) no contiene logica de modelado. Unicamente
  construye los formularios, envia peticiones HTTP al backend y muestra
  los resultados con tablas y graficas.
- El **backend** (FastAPI) recibe los datos en JSON, valida con Pydantic,
  consulta al modelo y devuelve la prediccion.
- El **modelo** (`model.pkl`) es un Random Forest serializado con joblib.

### Flujo de una prediccion individual

1. El usuario captura los datos del alumno en Streamlit y oprime el boton.
2. `frontend/app.py` arma un JSON con las 8 variables y hace
   `POST http://localhost:8000/predict`.
3. FastAPI valida los datos con Pydantic (`backend/schemas.py`). Si algun
   campo esta fuera de rango (por ejemplo `promedio_general > 10`),
   responde con codigo `422`.
4. `backend/predictor.py` convierte el diccionario a un DataFrame con las
   columnas en el orden esperado.
5. Llama a `modelo.predict_proba(df)` para obtener la probabilidad.
6. Calcula valores SHAP para identificar las variables que mas pesan en
   esta prediccion en particular.
7. Construye una explicacion textual y devuelve el JSON.
8. Streamlit dibuja: probabilidad, nivel de riesgo en color e impacto
   de las variables principales.

### Como se entrena el modelo (`model/train_model.py`)

1. **Genera 2000 alumnos sinteticos** con `numpy.random` usando
   distribuciones razonables (normal para promedio, poisson para materias
   no acreditadas, binomial para beca).
2. **Calcula un score de riesgo heuristico**:
   ```
   score = (10 - promedio_general) * 0.75
         + (100 - asistencias) * 0.04
         + materias_no_acreditadas * 0.80
         - horas_estudio * 0.08
         - beca * 0.60
         + ruido_aleatorio
   ```
3. **Define la variable objetivo**: los alumnos con score en el ~22 %
   mas alto reciben la etiqueta `abandono = 1` (cifra cercana a la tasa
   de abandono reportada en universidades publicas mexicanas).
4. **Particion estratificada** 80/20 train/test.
5. Entrena `RandomForestClassifier(n_estimators=200, max_depth=8,
   class_weight="balanced")` para compensar el desbalance.
6. Evalua con accuracy y ROC-AUC.
7. Persiste:
   - `model/model.pkl` (modelo entrenado)
   - `model/model_metadata.json` (metricas y feature importances)
   - `data/alumnos_sinteticos.csv` (los 2000 registros)
   - `data/muestra_alumnos.csv` (20 registros para probar la carga
     por lotes en la interfaz)

### Endpoints del backend

| Metodo | Ruta          | Funcion                                                          |
|--------|---------------|------------------------------------------------------------------|
| GET    | `/health`     | Reporta si el servicio esta vivo y el modelo cargado             |
| GET    | `/model/info` | Devuelve algoritmo, version, metricas y feature importances      |
| POST   | `/predict`    | Predice para un alumno o para un lote                            |

**Documentacion interactiva (Swagger)**: `http://localhost:8000/docs`.

### Explicabilidad con SHAP

Para cada prediccion individual se calculan valores SHAP que indican
**cuanto contribuye cada variable** al riesgo:

- Valor SHAP **positivo**: la variable AUMENTA el riesgo de abandono.
- Valor SHAP **negativo**: la variable REDUCE el riesgo.

Eso es lo que se muestra en la grafica horizontal verde/roja y en la
frase "Variables determinantes: ...".

Para lotes mayores a 50 alumnos se omite SHAP (costoso) y se usa una
aproximacion con la importancia global de cada variable.

---

## PARTE 4: SOLUCION DE PROBLEMAS

### "No se pudo conectar al backend"
Significa que el backend no esta corriendo o esta en otro puerto.
- Revisar la otra ventana de terminal: debe mostrar
  `Uvicorn running on http://0.0.0.0:8000`.
- Verificar con: `curl http://localhost:8000/health`.
- Si esta en otro puerto, definir antes de lanzar Streamlit:
  ```
  set API_URL=http://localhost:9000      (Windows)
  export API_URL=http://localhost:9000   (Linux/macOS)
  ```

### "Modelo no cargado" / archivo `.pkl` no encontrado
Falta entrenar. Ejecutar manualmente:
```
python model/train_model.py
```

### Error al instalar dependencias
- Actualizar pip: `python -m pip install --upgrade pip`.
- Si se usa Python 3.14 y alguna libreria falla al compilar, instalar
  Python 3.12 y recrear el venv:
  ```
  py -3.12 -m venv .venv
  ```

### Puerto 8000 u 8501 ocupado
Cambiar los puertos en `run.bat` / `run.sh`:
```
uvicorn backend.main:app --port 8001
streamlit run frontend/app.py --server.port 8502
```
(Y exportar `API_URL=http://localhost:8001` antes de lanzar Streamlit.)

### El navegador no abre automaticamente
Abrirlo a mano en `http://localhost:8501`.

### "ModuleNotFoundError: No module named 'backend'"
Uvicorn se esta ejecutando desde la carpeta equivocada. Hay que estar
parado en la raiz del proyecto (donde esta `requirements.txt`), no
dentro de `backend/`.

---

## PARTE 5: EJERCICIOS SUGERIDOS PARA EL TALLER

Ideas para extender el sistema durante la sesion:

1. **Anadir una nueva variable** (por ejemplo `division` con valores
   CBI/CBS/CSH, o `trabaja_medio_tiempo`):
   - Agregarla a `FEATURES` en `train_model.py` y `predictor.py`.
   - Anadirla al esquema `Alumno` en `schemas.py`.
   - Agregar el campo en el formulario de `frontend/app.py`.
   - Re-entrenar: borrar `model/model.pkl` y volver a ejecutar `run.bat`.
2. **Cambiar el algoritmo**: reemplazar `RandomForestClassifier` por
   `XGBClassifier` o `LogisticRegression` en `train_model.py`.
3. **Ajustar el umbral de riesgo**: editar `_nivel_riesgo` en
   `backend/predictor.py` (Bajo < 30 %, Medio < 60 %, Alto >= 60 %
   por defecto).
4. **Sustituir los datos sinteticos por datos reales** (anonimizados):
   modificar `generar_datos_sinteticos` en `train_model.py` para leer
   un CSV proporcionado por servicios escolares.
5. **Empaquetar para despliegue**: crear un `Dockerfile` por servicio
   (backend y frontend) y subirlo a un servidor de la universidad.
