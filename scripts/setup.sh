#!/bin/bash

# ===== PM2.5 MLP Dashboard - Environment Setup Script =====
# Este script configura el entorno de desarrollo/local

set -e  # Exit on any error

# ===== Colors =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===== Functions =====
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python() {
    log_info "Verificando Python..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 no está instalado"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_success "Python $PYTHON_VERSION encontrado"
    
    # Verificar versión mínima
    if [[ "$PYTHON_VERSION" < "3.12" ]]; then
        log_warning "Se recomienda Python 3.12.7 o superior"
    fi
}

create_venv() {
    log_info "Creando entorno virtual..."
    
    if [ -d "venv" ]; then
        log_warning "El entorno virtual ya existe, omitiendo creación"
        return 0
    fi
    
    python3 -m venv venv
    
    if [ $? -eq 0 ]; then
        log_success "Entorno virtual creado"
    else
        log_error "Error creando entorno virtual"
        exit 1
    fi
}

activate_venv() {
    log_info "Activando entorno virtual..."
    
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    if [ "$VIRTUAL_ENV" != "" ]; then
        log_success "Entorno virtual activado: $VIRTUAL_ENV"
    else
        log_error "No se pudo activar el entorno virtual"
        exit 1
    fi
}

install_dependencies() {
    log_info "Instalando dependencias..."
    
    # Actualizar pip
    python -m pip install --upgrade pip
    
    # Instalar dependencias
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Dependencias instaladas desde requirements.txt"
    else
        log_error "No se encontró requirements.txt"
        exit 1
    fi
}

setup_environment() {
    log_info "Configurando variables de entorno..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Archivo .env creado desde .env.example"
            log_warning "Por favor, edita .env con tus credenciales"
        else
            log_warning "No se encontró .env.example, creando .env básico"
            cat > .env << EOF
# Flask Configuration
FLASK_APP=app_flask.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Firebase Configuration (Optional)
FIREBASE_DB_URL=https://your-project.firebaseio.com
FIREBASE_API_KEY=your-firebase-api-key
FIREBASE_AUTH_EMAIL=your-email@domain.com
FIREBASE_AUTH_PASSWORD=your-firebase-password

# Model Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=50MB
EOF
            log_success "Archivo .env básico creado"
        fi
    else
        log_info "Archivo .env ya existe, omitiendo creación"
    fi
}

create_directories() {
    log_info "Creando directorios necesarios..."
    
    mkdir -p uploads
    mkdir -p logs
    mkdir -p models
    mkdir -p static/css
    mkdir -p static/js
    mkdir -p static/images
    mkdir -p tests
    
    log_success "Directorios creados"
}

setup_git_hooks() {
    log_info "Configurando Git hooks..."
    
    # Pre-commit hook para validación básica
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook básico

# Verificar sintaxis de Python
python -m py_compile app_flask.py
if [ $? -ne 0 ]; then
    echo "Error de sintaxis en app_flask.py"
    exit 1
fi

echo "✅ Validación de sintaxis exitosa"
EOF
    
    chmod +x .git/hooks/pre-commit
    log_success "Git hooks configurados"
}

run_tests() {
    log_info "Ejecutando pruebas básicas..."
    
    # Prueba de importación
    python -c "import flask; import tensorflow; import pandas; print('✅ Todas las dependencias se importan correctamente')"
    
    # Prueba de aplicación básica
    python -c "
from app_flask import app
with app.test_client() as client:
    response = client.get('/')
    assert response.status_code == 200
    print('✅ Aplicación Flask funciona correctamente')
"
    
    log_success "Pruebas básicas exitosas"
}

create_run_scripts() {
    log_info "Creando scripts de ejecución..."
    
    # Script para desarrollo
    cat > run_dev.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export FLASK_ENV=development
export FLASK_DEBUG=True
flask run
EOF
    
    # Script para producción (local)
    cat > run_prod.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export FLASK_ENV=production
gunicorn app_flask:app --bind 0.0.0.0:5000 --workers 4
EOF
    
    chmod +x run_dev.sh run_prod.sh
    log_success "Scripts de ejecución creados"
}

print_summary() {
    log_success "=== Setup Completo ==="
    echo ""
    echo "📌 Pasos siguientes:"
    echo "  1. Edita el archivo .env con tus credenciales"
    echo "  2. Activa el entorno virtual: source venv/bin/activate"
    echo "  3. Ejecuta la aplicación: ./run_dev.sh"
    echo "  4. Accede a: http://localhost:5000"
    echo ""
    echo "🛠️ Comandos útiles:"
    echo "  Activar entorno: source venv/bin/activate"
    echo "  Desactivar: deactivate"
    echo "  Instalar dependencias: pip install -r requirements.txt"
    echo "  Ejecutar tests: pytest"
    echo ""
    echo "📚 Documentación:"
    echo "  - README.md - Guía general del proyecto"
    echo "  - docs/deployment.md - Guía de despliegue"
    echo ""
}

# ===== Main Execution =====
main() {
    echo "🚀 Iniciando configuración del entorno..."
    echo ""
    
    check_python
    create_venv
    activate_venv
    install_dependencies
    setup_environment
    create_directories
    setup_git_hooks
    run_tests
    create_run_scripts
    
    print_summary
}

# ===== Error Handling =====
trap 'log_error "Setup fallido en la línea $LINENO"' ERR

# ===== Execute =====
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi