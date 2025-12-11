@echo off
REM ===== PM2.5 MLP Dashboard - Windows Setup Script =====
REM Este script configura el entorno de desarrollo en Windows

setlocal enabledelayedexpansion

REM ===== Colors (Basic) =====
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

echo.
echo 🚀 Iniciando configuración del entorno...
echo.

REM ===== Verificar Python =====
echo %INFO% Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python no está instalado o no está en el PATH
    echo Por favor, instala Python 3.12.7 desde https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %SUCCESS% Python !PYTHON_VERSION! encontrado

REM ===== Crear entorno virtual =====
echo.
echo %INFO% Creando entorno virtual...
if exist venv (
    echo %WARNING% El entorno virtual ya existe, omitiendo creación
) else (
    python -m venv venv
    if !errorlevel! equ 0 (
        echo %SUCCESS% Entorno virtual creado
    ) else (
        echo %ERROR% Error creando entorno virtual
        pause
        exit /b 1
    )
)

REM ===== Activar entorno virtual =====
echo.
echo %INFO% Activando entorno virtual...
call venv\Scripts\activate

if "!VIRTUAL_ENV!" neq "" (
    echo %SUCCESS% Entorno virtual activado
) else (
    echo %ERROR% No se pudo activar el entorno virtual
    pause
    exit /b 1
)

REM ===== Actualizar pip =====
echo.
echo %INFO% Actualizando pip...
python -m pip install --upgrade pip

REM ===== Instalar dependencias =====
echo.
echo %INFO% Instalando dependencias...
if exist requirements.txt (
    pip install -r requirements.txt
    if !errorlevel! equ 0 (
        echo %SUCCESS% Dependencias instaladas
    ) else (
        echo %ERROR% Error instalando dependencias
        pause
        exit /b 1
    )
) else (
    echo %ERROR% No se encontró requirements.txt
    pause
    exit /b 1
)

REM ===== Crear directorios =====
echo.
echo %INFO% Creando directorios necesarios...
if not exist uploads mkdir uploads
if not exist logs mkdir logs
if not exist models mkdir models
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js
if not exist static\images mkdir static\images
if not exist tests mkdir tests
echo %SUCCESS% Directorios creados

REM ===== Configurar variables de entorno =====
echo.
echo %INFO% Configurando variables de entorno...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo %SUCCESS% Archivo .env creado desde .env.example
        echo %WARNING% Por favor, edita .env con tus credenciales
    ) else (
        echo %WARNING% No se encontró .env.example, creando .env básico
        (
            echo # Flask Configuration
            echo FLASK_APP=app_flask.py
            echo FLASK_ENV=development
            echo FLASK_DEBUG=True
            echo SECRET_KEY=dev-secret-key-change-in-production
            echo.
            echo # Server Configuration
            echo HOST=0.0.0.0
            echo PORT=5000
            echo.
            echo # Firebase Configuration (Optional)
            echo FIREBASE_DB_URL=https://your-project.firebaseio.com
            echo FIREBASE_API_KEY=your-firebase-api-key
            echo FIREBASE_AUTH_EMAIL=your-email@domain.com
            echo FIREBASE_AUTH_PASSWORD=your-firebase-password
            echo.
            echo # Model Configuration
            echo UPLOAD_FOLDER=uploads
            echo MAX_CONTENT_LENGTH=50MB
        ) > .env
        echo %SUCCESS% Archivo .env básico creado
    )
) else (
    echo %INFO% Archivo .env ya existe, omitiendo creación
)

REM ===== Crear scripts de ejecución =====
echo.
echo %INFO% Creando scripts de ejecución...

REM Script para desarrollo
(
    echo @echo off
    echo call venv\\Scripts\\activate
    echo set FLASK_ENV=development
    echo set FLASK_DEBUG=True
    echo flask run
    echo pause
) > run_dev.bat

REM Script para producción (local)
(
    echo @echo off
    echo call venv\\Scripts\\activate
    echo set FLASK_ENV=production
    echo gunicorn app_flask:app --bind 0.0.0.0:5000 --workers 4
    echo pause
) > run_prod.bat

echo %SUCCESS% Scripts de ejecución creados

REM ===== Pruebas básicas =====
echo.
echo %INFO% Ejecutando pruebas básicas...

REM Prueba de importación
python -c "import flask; import tensorflow; import pandas; print('✅ Todas las dependencias se importan correctamente')"

if !errorlevel! equ 0 (
    echo %SUCCESS% Prueba de importación exitosa
) else (
    echo %ERROR% Error en pruebas de importación
    pause
    exit /b 1
)

REM ===== Resumen =====
echo.
echo.
echo %SUCCESS% === Setup Completo ===
echo.
echo 📌 Pasos siguientes:
echo   1. Edita el archivo .env con tus credenciales
echo   2. Activa el entorno virtual: venv\Scripts\activate
echo   3. Ejecuta la aplicación: run_dev.bat
echo   4. Accede a: http://localhost:5000
echo.
echo 🛠️ Comandos útiles:
echo   Activar entorno: venv\Scripts\activate
echo   Desactivar: deactivate
echo   Instalar dependencias: pip install -r requirements.txt
echo   Ejecutar tests: pytest
echo.
echo 📚 Documentación:
echo   - README.md - Guía general del proyecto
echo   - docs\deployment.md - Guía de despliegue
echo.
echo ✅ Configuración completada exitosamente!
pause