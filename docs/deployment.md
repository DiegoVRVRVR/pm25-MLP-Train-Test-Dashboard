# 🚀 Guía de Despliegue

Esta guía detalla cómo desplegar la aplicación en diferentes entornos.

## 📋 Requisitos Previos

- Cuenta en [Render](https://render.com)
- Repositorio en [GitHub](https://github.com)
- Credenciales de Firebase (opcional)

## ☁️ Despliegue en Render

### Paso 1: Preparación del Repositorio

1. **Crear repositorio en GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/tu-usuario/daily_model_interface.git
   git push -u origin main
   ```

2. **Verificar archivos esenciales**
   - `Procfile` - Configuración del servicio
   - `runtime.txt` - Versión de Python
   - `requirements.txt` - Dependencias
   - `app_flask.py` - Aplicación principal

### Paso 2: Configuración en Render

1. **Iniciar sesión en Render**
   - Visita https://dashboard.render.com
   - Haz clic en "New Web Service"

2. **Conectar GitHub**
   - Selecciona tu repositorio
   - Elige la rama `main`

3. **Configuración del Servicio**
   ```
   Name: daily-model-interface
   Branch: main
   Region: oregon (recomendado)
   ```

4. **Build Settings**
   ```
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app_flask:app --bind 0.0.0.0:$PORT
   ```

5. **Environment**
   ```
   Environment: Python
   Python Version: 3.12.7
   ```

### Paso 3: Variables de Entorno

Agrega estas variables en la sección "Environment" de Render:

```env
FLASK_ENV=production
SECRET_KEY=tu-clave-secreta-muy-segura
PORT=10000
```

**Para generar una clave secreta segura:**
```python
import secrets
print(secrets.token_hex(32))
```

### Paso 4: Firebase (Opcional)

Si deseas habilitar el despliegue automático de modelos:

```env
FIREBASE_DB_URL=https://tu-proyecto.firebaseio.com
FIREBASE_API_KEY=tu-api-key
FIREBASE_AUTH_EMAIL=tu-email@dominio.com
FIREBASE_AUTH_PASSWORD=tu-contraseña-segura
```

### Paso 5: Desplegar

1. Haz clic en "Create Web Service"
2. Espera 5-10 minutos a que termine el despliegue
3. Render proporcionará una URL como: `https://daily-model-interface.onrender.com`

## 🐳 Despliegue con Docker

### Opción A: Docker Local

```bash
# Construir la imagen
docker build -t daily-model-interface .

# Ejecutar el contenedor
docker run -p 5000:5000 daily-model-interface

# Ver logs
docker logs -f <container_id>
```

### Opción B: Docker Compose

Crea un archivo `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=tu-clave-secreta
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
```

Ejecuta:
```bash
docker-compose up -d
```

### Opción C: Render con Docker

1. En Render, selecciona "New Web Service"
2. Elige "Docker"
3. Conecta tu repositorio
4. Configura:
   ```
   Docker Image: diegolvrvr/daily-model-interface:latest
   Port: 5000
   ```

## 🔍 Verificación del Despliegue

### Health Check

Después del despliegue, verifica que todo funcione:

```bash
# Verifica el endpoint de salud
curl https://tu-dominio.onrender.com/health

# Debe retornar: {"status": "healthy", "version": "1.0.0"}
```

### Pruebas Básicas

1. **Accede al dashboard**: https://tu-dominio.onrender.com
2. **Verifica carga de archivos**: Sube un CSV de prueba
3. **Prueba entrenamiento**: Configura un modelo rápido (pocas épocas)
4. **Verifica Firebase**: Si está configurado, intenta desplegar un modelo

## 📊 Monitoreo y Logs

### En Render

1. Ve a tu servicio en Render Dashboard
2. Sección "Logs" para ver actividad en tiempo real
3. Sección "Metrics" para métricas de CPU, memoria, etc.

### Comandos Útiles

```bash
# Ver logs en tiempo real
render logs -s <service-id>

# Reiniciar servicio
render restart -s <service-id>

# Ver estado
render status -s <service-id>
```

## 🔧 Configuración Avanzada

### SSL/HTTPS

Render maneja SSL automáticamente:
- Certificados SSL gratuitos con Let's Encrypt
- Redirección automática HTTP → HTTPS
- No requiere configuración adicional

### Dominio Personalizado

1. En Render, ve a tu servicio
2. Sección "Custom Domains"
3. Agrega tu dominio: `app.tu-dominio.com`
4. Configura DNS en tu proveedor:
   ```
   CNAME: app.tu-dominio.com → tu-servicio.onrender.com
   ```

### Escalado

Para alto tráfico:

1. **Horizontal**: Aumenta instancias en Render
2. **Vertical**: Aumenta RAM/CPU
3. **Cache**: Configura Redis para sesiones

### Base de Datos

Para producción, considera:

```env
# PostgreSQL en Render
DATABASE_URL=postgresql://user:pass@host:port/db
```

## 🐛 Solución de Problemas

### Errores Comunes

**Error 500 - Internal Server Error**
```bash
# Ver logs detallados
render logs -s <service-id> --tail 100

# Verifica variables de entorno
# Revisa requirements.txt
# Confirma que app_flask.py tenga la app Flask
```

**Error de Build**
```bash
# Verifica runtime.txt
# Confirma Python 3.12.7
# Revisa dependencias en requirements.txt
```

**Firebase Connection Failed**
```bash
# Verifica credenciales
# Confirma reglas de seguridad en Firebase
# Prueba conexión localmente primero
```

### Comandos de Depuración

```bash
# Prueba local antes de desplegar
flask run --debug

# Verifica dependencias
pip check

# Prueba Docker localmente
docker run --rm -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=test \
  daily-model-interface
```

## 📈 Mejores Prácticas

### Seguridad

- ✅ Usa HTTPS siempre
- ✅ Claves secretas únicas y largas
- ✅ No expongas información sensible en logs
- ✅ Limita tamaño de archivos subidos
- ✅ Valida todos los inputs del usuario

### Performance

- ✅ Usa Gunicorn con workers apropiados
- ✅ Configura timeouts adecuados
- ✅ Implementa cache cuando sea posible
- ✅ Optimiza imágenes y assets estáticos
- ✅ Usa CDN para recursos estáticos

### Mantenimiento

- ✅ Monitorea logs regularmente
- ✅ Actualiza dependencias periódicamente
- ✅ Haz backup de datos importantes
- ✅ Prueba despliegues en staging primero
- ✅ Documenta cambios y configuraciones

## 📞 Soporte

Si tienes problemas:

1. Revisa los logs en Render Dashboard
2. Verifica esta documentación
3. Abre un issue en GitHub
4. Contacta al equipo de desarrollo

---

**Nota**: Esta guía asume conocimientos básicos de Git, GitHub y Render. Para más información, consulta la [documentación oficial de Render](https://render.com/docs).