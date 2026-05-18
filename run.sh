#!/usr/bin/env bash
# run.sh - Levanta el proyecto en Linux / macOS.
# Pasos:
#   1) Crea (si no existe) un venv en .venv
#   2) Instala dependencias
#   3) Entrena el modelo si no existe model/model.pkl
#   4) Levanta el backend (FastAPI) en segundo plano
#   5) Levanta el frontend (Streamlit) en primer plano
# Para detener: Ctrl+C en el frontend; el backend se cierra al salir el script.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# 1) Crear venv si no existe.
if [ ! -d ".venv" ]; then
    echo ">> Creando entorno virtual en .venv ..."
    "$PYTHON_BIN" -m venv .venv
fi

# Activar venv.
# shellcheck disable=SC1091
source .venv/bin/activate

# 2) Instalar dependencias.
echo ">> Instalando dependencias ..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# 3) Entrenar modelo si no existe.
if [ ! -f "model/model.pkl" ]; then
    echo ">> Entrenando modelo (primera ejecucion) ..."
    python model/train_model.py
else
    echo ">> Modelo ya existe en model/model.pkl. Salto entrenamiento."
fi

# 4) Levantar backend en segundo plano.
echo ">> Levantando backend en http://localhost:8000 ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Asegurar cierre del backend al terminar el script.
trap "echo '>> Cerrando backend (PID $BACKEND_PID) ...'; kill $BACKEND_PID 2>/dev/null || true" EXIT

# Pequena pausa para que arranque el backend.
sleep 3

# 5) Levantar frontend.
echo ">> Levantando frontend en http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501
