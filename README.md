# Prediccion de Abandono Estudiantil - UAM Iztapalapa

Sistema para estimar el riesgo de abandono escolar de los alumnos de la
**Universidad Autonoma Metropolitana, Unidad Iztapalapa (UAM-I)** a
partir de su historial academico.

Componentes:

- **Backend** en FastAPI con un modelo Random Forest.
- **Frontend** en Streamlit (captura individual + carga por lotes).
- Explicabilidad por SHAP para identificar las variables que mas pesan
  en cada prediccion.

Pensado como material didactico para un taller de 2 horas. Requiere
**Python 3.9 o superior** (se recomienda 3.11 o 3.12).

---

## Como ejecutar (5 pasos)

1. Copiar la carpeta `student-dropout-predictor` a tu equipo.
2. Abrir una terminal dentro de esa carpeta.
3. Ejecutar el script segun el sistema operativo:
   - Windows: `run.bat`
   - Linux / macOS: `bash run.sh`
4. Esperar a que termine la instalacion y el entrenamiento (la primera
   vez tarda 1 a 2 minutos).
5. Abrir el navegador en **http://localhost:8501**.

El script crea el entorno virtual, instala dependencias, entrena el
modelo (la primera vez), levanta el backend en `http://localhost:8000`
y el frontend en `http://localhost:8501`.

---

## Estructura del proyecto

```
student-dropout-predictor/
|-- backend/
|   |-- main.py          API FastAPI (/predict, /health, /model/info)
|   |-- predictor.py     Carga del modelo, SHAP y explicaciones
|   |-- schemas.py       Esquemas Pydantic (validacion de datos)
|-- frontend/
|   |-- app.py           Interfaz Streamlit
|-- model/
|   |-- train_model.py   Generacion de datos sinteticos y entrenamiento
|   |-- model.pkl                (se crea al entrenar)
|   |-- model_metadata.json      (se crea al entrenar)
|-- data/
|   |-- alumnos_sinteticos.csv   (se crea al entrenar)
|   |-- muestra_alumnos.csv      (se crea al entrenar)
|-- requirements.txt
|-- run.bat / run.sh
|-- README.md
|-- INSTRUCCIONES.md
```

---

## Endpoints del backend

| Metodo | Ruta          | Descripcion                                       |
|--------|---------------|---------------------------------------------------|
| GET    | `/health`     | Verifica el estado del servicio y del modelo      |
| GET    | `/model/info` | Metadata: features, metricas, importancias        |
| POST   | `/predict`    | Prediccion individual o por lote                  |

Documentacion interactiva: **http://localhost:8000/docs**

### Ejemplo: prediccion individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 21,
    "trimestre": 4,
    "promedio_general": 6.5,
    "asistencias_porcentaje": 70,
    "materias_no_acreditadas": 3,
    "creditos_acumulados": 120,
    "horas_estudio_semana": 5,
    "beca": 0
  }'
```

### Ejemplo: prediccion por lote

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "alumnos": [ {...}, {...} ] }'
```

---

## Variables que utiliza el modelo

Las variables siguen la nomenclatura de los sistemas escolares de la UAM-I:

| Variable                   | Tipo  | Rango / valores            |
|----------------------------|-------|----------------------------|
| `edad`                     | int   | 15 - 80                    |
| `trimestre`                | int   | 1 - 12                     |
| `promedio_general`         | float | 0.0 - 10.0                 |
| `asistencias_porcentaje`   | float | 0.0 - 100.0                |
| `materias_no_acreditadas`  | int   | 0 - 20 (UEAS marcadas NA)  |
| `creditos_acumulados`      | int   | 0 - 500                    |
| `horas_estudio_semana`     | float | 0.0 - 80.0                 |
| `beca`                     | int   | 0 = No, 1 = Si             |

Variable objetivo: `abandono` (0 = continua, 1 = abandona).

> Aviso: el dataset utilizado para el entrenamiento es **sintetico**. No
> proviene de datos reales del Sistema de Administracion Escolar (SAE)
> ni de ningun sistema de la UAM-I. Sirve exclusivamente como material
> didactico para el taller.

---

## Solucion de problemas

**No se pudo conectar al backend**  
Verifica que la API este corriendo en `http://localhost:8000`. Si la
levantaste en otro puerto, define la variable de entorno antes de
lanzar Streamlit:

```
set API_URL=http://localhost:9000      (Windows)
export API_URL=http://localhost:9000   (Linux/macOS)
```

**Modelo no cargado**  
Ejecuta manualmente: `python model/train_model.py`.

**Permisos en `run.sh`**  
`chmod +x run.sh` y luego `./run.sh` (o `bash run.sh`).

**Puerto ocupado**  
Cambia el puerto en `run.bat` / `run.sh` (uvicorn `--port` y streamlit
`--server.port`).
