# ===== Configuración de Producción =====
# Esta configuración es para entornos de producción en Render

import os
from datetime import timedelta

# ===== Flask Configuration =====
DEBUG = False
TESTING = False

# ===== Server Configuration =====
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 10000))

# ===== Security Configuration =====
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY es requerida en producción")

# ===== Session Configuration =====
SESSION_COOKIE_SECURE = True  # Solo HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)

# ===== CSRF Protection =====
WTF_CSRF_ENABLED = True
WTF_CSRF_TIME_LIMIT = None

# ===== Database Configuration =====
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')

# ===== Model Configuration =====
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024)  # 50MB
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_mlp.tflite')

# ===== Logging Configuration =====
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')

# ===== Firebase Configuration =====
FIREBASE_DB_URL = os.environ.get('FIREBASE_DB_URL')
FIREBASE_API_KEY = os.environ.get('FIREBASE_API_KEY')
FIREBASE_AUTH_EMAIL = os.environ.get('FIREBASE_AUTH_EMAIL')
FIREBASE_AUTH_PASSWORD = os.environ.get('FIREBASE_AUTH_PASSWORD')

# ===== External APIs =====
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# ===== Performance Configuration =====
# Gunicorn settings (para Render)
WORKERS = int(os.environ.get('GUNICORN_WORKERS', 4))
WORKER_CLASS = os.environ.get('GUNICORN_WORKER_CLASS', 'sync')
WORKER_CONNECTIONS = int(os.environ.get('GUNICORN_WORKER_CONNECTIONS', 1000))
TIMEOUT = int(os.environ.get('GUNICORN_TIMEOUT', 120))
KEEP_ALIVE = int(os.environ.get('GUNICORN_KEEP_ALIVE', 5))
MAX_REQUESTS = int(os.environ.get('GUNICORN_MAX_REQUESTS', 1000))
MAX_REQUESTS_JITTER = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 50))

# ===== Cache Configuration =====
CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 300))

# ===== Monitoring Configuration =====
SENTRY_DSN = os.environ.get('SENTRY_DSN')
PROMETHEUS_ENABLED = os.environ.get('PROMETHEUS_ENABLED', 'false').lower() == 'true'

# ===== CORS Configuration =====
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization']
CORS_ALLOW_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']

# ===== Health Check Configuration =====
HEALTH_CHECK_ENABLED = True
HEALTH_CHECK_PATH = '/health'

# ===== Model Training Configuration =====
DEFAULT_EPOCHS = int(os.environ.get('DEFAULT_EPOCHS', 200))
DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', 32))
DEFAULT_TEST_SIZE = float(os.environ.get('DEFAULT_TEST_SIZE', 0.2))
DEFAULT_MAX_LAG = int(os.environ.get('DEFAULT_MAX_LAG', 5))
DEFAULT_TUNING_TIME = int(os.environ.get('DEFAULT_TUNING_TIME', 5))

# ===== File Upload Configuration =====
ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ===== Email Configuration (for alerts) =====
MAIL_SERVER = os.environ.get('MAIL_SERVER')
MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

# ===== Development Override (for testing) =====
if os.environ.get('FLASK_ENV') == 'development':
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = 'DEBUG'