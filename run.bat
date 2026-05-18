@echo off
REM run.bat - Levanta el proyecto en Windows.
REM Pasos:
REM   1) Crea el venv si no existe
REM   2) Instala dependencias
REM   3) Entrena el modelo si no existe model\model.pkl
REM   4) Levanta el backend FastAPI en una ventana nueva
REM   5) Levanta el frontend Streamlit en esta ventana
REM Para detener: cierra ambas ventanas o presiona Ctrl+C.

setlocal
cd /d "%~dp0"

REM ---- 0) Detectar version de Python ----
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo ^>^> Python detectado: %PYVER%

echo %PYVER% | findstr /r "^3\.1[3-9]" >nul
if errorlevel 1 goto :crear_venv
echo [AVISO] Estas usando Python %PYVER%. Si la instalacion falla, instala
echo         Python 3.12 y recrea el venv con:  py -3.12 -m venv .venv
echo.

REM ---- 1) Crear venv si no existe ----
:crear_venv
if exist ".venv" goto :instalar_deps
echo ^>^> Creando entorno virtual en .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el venv. Verifica que Python 3.9+ este instalado.
    exit /b 1
)

REM ---- 2) Activar venv e instalar dependencias ----
:instalar_deps
call .venv\Scripts\activate.bat
echo ^>^> Instalando dependencias ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    exit /b 1
)

REM ---- 3) Entrenar modelo si no existe ----
if exist "model\model.pkl" goto :modelo_listo
echo ^>^> Entrenando modelo por primera vez ...
python model\train_model.py
if errorlevel 1 (
    echo [ERROR] Fallo el entrenamiento del modelo.
    exit /b 1
)
goto :levantar_backend

:modelo_listo
echo ^>^> Modelo ya existe en model\model.pkl. Salto entrenamiento.

REM ---- 4) Levantar backend en ventana nueva ----
:levantar_backend
echo ^>^> Levantando backend en http://localhost:8000 ...
start "API UAM-I Prediccion Abandono" cmd /k ".venv\Scripts\activate.bat && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Pausa para que arranque el backend.
timeout /t 4 /nobreak >nul

REM ---- 5) Levantar frontend ----
echo ^>^> Levantando frontend en http://localhost:8501 ...
streamlit run frontend\app.py --server.port 8501

endlocal
