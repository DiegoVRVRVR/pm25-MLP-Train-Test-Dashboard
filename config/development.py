# ===== Configuración de Desarrollo =====
# Esta configuración es para entornos de desarrollo local

import os
from datetime import timedelta

# ===== Flask Configuration =====
DEBUG = True
TESTING = False

# ===== Server Configuration =====
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', 5000))

# ===== Security Configuration =====
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ===== Session Configuration =====
SESSION_COOKIE_SECURE = False  # No HTTPS en desarrollo
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

# ===== CSRF Protection =====
WTF_CSRF_ENABLED = False  # Desactivado en desarrollo para pruebas

# ===== Database Configuration =====
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///dev_app.db')

# ===== Model Configuration =====
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024)  # 50MB
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_mlp.tflite')

# ===== Logging Configuration =====
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')
LOG_FILE = os.environ.get('LOG_FILE', 'logs/dev_app.log')

# ===== Firebase Configuration =====
FIREBASE_DB_URL = os.environ.get('FIREBASE_DB_URL', 'https://esp32-pms7003-database-system-default-rtdb.firebaseio.com')
FIREBASE_API_KEY = os.environ.get('FIREBASE_API_KEY', 'AIzaSyBu-RdQfglvyc9DNFARIB9XwOnUQwtPI5A')
FIREBASE_AUTH_EMAIL = os.environ.get('FIREBASE_AUTH_EMAIL', 'admin@esp32ml.com')
FIREBASE_AUTH_PASSWORD = os.environ.get('FIREBASE_AUTH_PASSWORD', 'MiPassword123!')

# ===== External APIs =====
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# ===== Performance Configuration =====
# Configuración más ligera para desarrollo
WORKERS = 1
WORKER_CLASS = 'sync'
WORKER_CONNECTIONS = 100
TIMEOUT = 300
KEEP_ALIVE = 30
MAX_REQUESTS = 100
MAX_REQUESTS_JITTER = 10

# ===== Cache Configuration =====
CACHE_TYPE = 'null'  # Desactivado en desarrollo
CACHE_DEFAULT_TIMEOUT = 300

# ===== Monitoring Configuration =====
SENTRY_DSN = None  # Desactivado en desarrollo
PROMETHEUS_ENABLED = False

# ===== CORS Configuration =====
CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization', 'X-Requested-With']
CORS_ALLOW_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD']

# ===== Health Check Configuration =====
HEALTH_CHECK_ENABLED = True
HEALTH_CHECK_PATH = '/health'

# ===== Model Training Configuration =====
DEFAULT_EPOCHS = int(os.environ.get('DEFAULT_EPOCHS', 50))  # Menos épocas para desarrollo
DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', 16))
DEFAULT_TEST_SIZE = float(os.environ.get('DEFAULT_TEST_SIZE', 0.3))
DEFAULT_MAX_LAG = int(os.environ.get('DEFAULT_MAX_LAG', 3))
DEFAULT_TUNING_TIME = int(os.environ.get('DEFAULT_TUNING_TIME', 2))

# ===== File Upload Configuration =====
ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx', 'xls'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB en desarrollo

# ===== Email Configuration (for alerts) =====
MAIL_SERVER = os.environ.get('MAIL_SERVER', 'localhost')
MAIL_PORT = int(os.environ.get('MAIL_PORT', 1025))  # Puerto para desarrollo
MAIL_USE_TLS = False  # Desactivado en desarrollo
MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'dev@localhost')

# ===== Development Tools =====
# Configuración para herramientas de desarrollo
USE_PROFILER = os.environ.get('USE_PROFILER', 'false').lower() == 'true'
USE_DEBUG_TOOLBAR = os.environ.get('USE_DEBUG_TOOLBAR', 'true').lower() == 'true'

# ===== Testing Configuration =====
TESTING = os.environ.get('TESTING', 'false').lower() == 'true'
TEST_DATABASE_URL = 'sqlite:///:memory:'