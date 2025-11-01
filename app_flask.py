from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import io
import base64
import hashlib
import datetime
import requests
import itertools
import time
import json
import os
import warnings
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

plt.style.use('ggplot')
sns.set_palette("husl")

# ==================== CONFIGURACIÓN FIREBASE ====================
FIREBASE_DB_URL = "https://esp32-pms7003-database-system-default-rtdb.firebaseio.com"
FIREBASE_API_KEY = "AIzaSyBu-RdQfglvyc9DNFARIB9XwOnUQwtPI5A"
FIREBASE_AUTH_EMAIL = "admin@esp32ml.com"
FIREBASE_AUTH_PASSWORD = "MiPassword123!"

def get_firebase_token():
    """Obtiene un token de autenticación de Firebase"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": FIREBASE_AUTH_EMAIL,
        "password": FIREBASE_AUTH_PASSWORD,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()['idToken']
        else:
            print(f"Error obteniendo token: {response.text}")
            return None
    except Exception as e:
        print(f"Excepción obteniendo token: {e}")
        return None

# ==================== FUNCIONES DEL MODELO ====================

def load_and_combine_data(file_paths):
    """Carga y combina archivos CSV - MANTIENE SIEMPRE LAS 3 VARIABLES"""
    try:
        dfs = []
        files_data = [
            (file_paths.get('pm1'), 'PM1'),
            (file_paths.get('pm25'), 'PM25'),
            (file_paths.get('pm10'), 'PM10')
        ]
        
        print(f"[DEBUG] Iniciando carga de archivos CSV...")
        
        for file_path, col_name in files_data:
            if file_path and os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=0)
                    print(f"[DEBUG] {col_name} - Shape original: {df.shape}")
                    
                    df.set_index(df.columns[0], inplace=True)
                    df.index.name = 'FECHA_HORA'
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df.columns = [col_name]
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    df = df[df.index.notna()]
                    
                    print(f"[DEBUG] {col_name} - NaN: {df[col_name].isna().sum()}/{len(df)}")
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"[ERROR] Cargando {col_name}: {str(e)}")
                    continue
        
        if len(dfs) == 0:
            return None
        
        combined_df = pd.concat(dfs, axis=1, join='outer')
        print(f"[DEBUG] Combinado - Shape: {combined_df.shape}, Columnas: {combined_df.columns.tolist()}")
        
        daily_df = combined_df.resample('D').mean()
        print(f"[DEBUG] Después resample - Shape: {daily_df.shape}")
        
        # RELLENAR NaN INTELIGENTEMENTE - MANTENER SIEMPRE LAS 3 COLUMNAS
        for col in daily_df.columns:
            nan_before = daily_df[col].isna().sum()
            
            daily_df[col] = daily_df[col].interpolate(method='linear', limit_direction='both')
            daily_df[col] = daily_df[col].fillna(method='ffill', limit=7)
            daily_df[col] = daily_df[col].fillna(method='bfill', limit=7)
            
            if daily_df[col].isna().any():
                daily_df[col] = daily_df[col].fillna(daily_df[col].mean())
            
            if daily_df[col].isna().any():
                daily_df[col] = daily_df[col].fillna(daily_df.mean(axis=1))
            
            daily_df[col] = daily_df[col].fillna(0)
            
            nan_after = daily_df[col].isna().sum()
            print(f"[DEBUG] {col} - NaN rellenados: {nan_before} → {nan_after}")
        
        print(f"[DEBUG] Final - Shape: {daily_df.shape}, NaN total: {daily_df.isna().sum().sum()}")
        return daily_df
    
    except Exception as e:
        print(f"[ERROR] load_and_combine_data: {str(e)}")
        traceback.print_exc()
        return None


def create_time_series_features(df, target_col='PM25', max_lag=5):
    """Crea características temporales"""
    try:
        features_df = pd.DataFrame(index=df.index)
        
        if target_col not in df.columns:
            print(f"[ERROR] {target_col} no encontrado. Columnas: {df.columns.tolist()}")
            return None
        
        features_df['target'] = df[target_col].shift(-1)
        
        base_vars = ['PM25', 'PM10', 'PM1']
        for var in base_vars:
            if var in df.columns:
                for lag in range(1, max_lag + 1):
                    features_df[f'{var}_lag_{lag}'] = df[var].shift(lag)
        
        if target_col in df.columns:
            daily_means = df.groupby(df.index.dayofweek)[target_col].mean()
            features_df['pm25_weekday_avg'] = features_df.index.dayofweek.map(daily_means)
        
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['day_of_year'] = features_df.index.dayofyear
        features_df['days_from_start'] = (features_df.index - features_df.index.min()).days
        
        return features_df
    
    except Exception as e:
        print(f"[ERROR] create_time_series_features: {str(e)}")
        traceback.print_exc()
        return None


def preprocess_data(features_df):
    """Preprocesa datos - MANTIENE TODAS LAS COLUMNAS"""
    try:
        print(f"[DEBUG] Preprocess - Shape inicial: {features_df.shape}")
        
        features_df = features_df.dropna(how='all')
        
        for col in features_df.columns:
            if features_df[col].isna().any():
                features_df[col] = features_df[col].interpolate(method='linear', limit_direction='both')
                features_df[col] = features_df[col].fillna(method='ffill').fillna(method='bfill')
                features_df[col] = features_df[col].fillna(features_df[col].mean())
                features_df[col] = features_df[col].fillna(0)
        
        if len(features_df) > 100:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                features_df[col] = features_df[col].rolling(window=3, min_periods=1, center=True).mean()
        
        nan_total = features_df.isna().sum().sum()
        print(f"[DEBUG] Preprocess - Shape final: {features_df.shape}, NaN: {nan_total}")
        
        if nan_total > 0:
            features_df = features_df.fillna(0)
        
        return features_df
    
    except Exception as e:
        print(f"[ERROR] preprocess_data: {str(e)}")
        traceback.print_exc()
        return None


def tune_mlp_architecture(X_train_scaled, y_train, X_test_scaled, y_test, max_time_minutes=10):
    """Búsqueda automática de arquitectura"""
    try:
        start_time = time.time()
        hidden_layers = [1, 2, 3]
        neurons_options = [4, 8, 16, 32]
        
        results = []
        best_config = None
        best_score = float('inf')
        
        for n_layers in hidden_layers:
            for neurons_combo in itertools.combinations_with_replacement(neurons_options, n_layers):
                if (time.time() - start_time) > (max_time_minutes * 60):
                    break
                
                layers = []
                for i, n_neurons in enumerate(neurons_combo):
                    if i == 0:
                        layers.append(tf.keras.layers.Dense(
                            n_neurons, activation='relu',
                            input_shape=(X_train_scaled.shape[1],)
                        ))
                    else:
                        layers.append(tf.keras.layers.Dense(n_neurons, activation='relu'))
                layers.append(tf.keras.layers.Dense(1, activation='linear'))
                
                model = tf.keras.Sequential(layers)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test_scaled, y_test),
                    verbose=0
                )
                
                val_mse = history.history['val_loss'][-1]
                val_mae = history.history['val_mae'][-1]
                n_params = model.count_params()
                
                config = {
                    'layers': list(neurons_combo),
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'params': n_params,
                    'score': val_mse + 0.001 * n_params
                }
                results.append(config)
                
                if config['score'] < best_score:
                    best_score = config['score']
                    best_config = config
        
        results.sort(key=lambda x: x['score'])
        return best_config, results[:5]
    
    except Exception as e:
        print(f"[ERROR] tune_mlp_architecture: {str(e)}")
        traceback.print_exc()
        return None, []


def build_model(config, input_shape):
    """Construye modelo"""
    try:
        layers = []
        
        for i, layer_cfg in enumerate(config):
            neurons = layer_cfg['neurons']
            activation = layer_cfg.get('activation', 'relu')
            
            if i == 0:
                layers.append(tf.keras.layers.Dense(
                    neurons, activation=activation,
                    input_shape=(input_shape,)
                ))
            else:
                layers.append(tf.keras.layers.Dense(neurons, activation=activation))
        
        layers.append(tf.keras.layers.Dense(1, activation='linear'))
        
        model = tf.keras.Sequential(layers)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    except Exception as e:
        print(f"[ERROR] build_model: {str(e)}")
        traceback.print_exc()
        return None
    


def train_model(file_paths, config, mode='auto'):
    """Entrena el modelo completo"""
    try:
        print(f"[DEBUG] Entrenamiento iniciado - modo: {mode}")
        print(f"[DEBUG] Config: {config}")
        
        daily_df = load_and_combine_data(file_paths)
        if daily_df is None or len(daily_df) == 0:
            return {'error': 'No se pudieron cargar los datos'}
        
        print(f"[DEBUG] Datos cargados: {daily_df.shape}")
        
        features_df = create_time_series_features(daily_df, max_lag=config.get('max_lag', 5))
        if features_df is None or len(features_df) == 0:
            return {'error': 'Error creando características'}
        
        print(f"[DEBUG] Features: {features_df.shape}")
        
        features_df = preprocess_data(features_df)
        if features_df is None or len(features_df) < 50:
            return {'error': f'Datos insuficientes: {len(features_df) if features_df is not None else 0}'}
        
        print(f"[DEBUG] Datos preprocesados: {features_df.shape}")
        
        X = features_df.drop('target', axis=1)
        y = features_df['target']
        
        if X.isnull().any().any() or y.isnull().any():
            return {'error': 'Hay valores NaN en los datos finales'}
        
        test_size = config.get('test_size', 0.2)
        split_idx = max(int(len(features_df) * (1 - test_size)), len(features_df) - 50)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"[DEBUG] Split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if mode == 'auto':
            print(f"[DEBUG] Auto-tuning...")
            best_config, top_configs = tune_mlp_architecture(
                X_train_scaled, y_train.values, X_test_scaled, y_test.values,
                max_time_minutes=config.get('tuning_time', 5)
            )
            if best_config is None:
                return {'error': 'Error en auto-tuning'}
            
            print(f"[DEBUG] Mejor config: {best_config}")
            layers_config = [{'neurons': n, 'activation': 'relu'} for n in best_config['layers']]
            model = build_model(layers_config, len(X.columns))
        else:
            print(f"[DEBUG] Modelo manual...")
            model = build_model(config['layers'], len(X.columns))
        
        if model is None:
            return {'error': 'Error construyendo el modelo'}
        
        print(f"[DEBUG] Entrenando modelo final...")
        history = model.fit(
            X_train_scaled, y_train.values,
            epochs=config.get('epochs', 200),
            batch_size=config.get('batch_size', 32),
            validation_data=(X_test_scaled, y_test.values),
            verbose=0
        )
        
        y_pred_train = model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
        
        metrics = {
            'train_mse': float(mean_squared_error(y_train.values, y_pred_train)),
            'test_mse': float(mean_squared_error(y_test.values, y_pred_test)),
            'train_mae': float(mean_absolute_error(y_train.values, y_pred_train)),
            'test_mae': float(mean_absolute_error(y_test.values, y_pred_test)),
            'train_r2': float(r2_score(y_train.values, y_pred_train)),
            'test_r2': float(r2_score(y_test.values, y_pred_test)),
            'n_params': int(model.count_params())
        }
        
        print(f"[DEBUG] Métricas: {metrics}")
        
        model._scaler = scaler
        model._input_features = list(X.columns)
        
        return {
            'model': model,
            'metrics': metrics,
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'predictions': {
                'y_test': y_test.values.tolist(),
                'y_pred': y_pred_test.tolist(),
                'dates': y_test.index.strftime('%Y-%m-%d').tolist()
            }
        }
    
    except Exception as e:
        print(f"[ERROR] train_model: {str(e)}")
        traceback.print_exc()
        return {'error': f'Error: {str(e)}'}


def upload_to_firebase(model_path, scaler_mean, scaler_scale, max_lag, model_config, use_auth=True):
    """Sube modelo a Firebase CON METADATA COMPLETA"""
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        model_hash = hashlib.md5(model_data).hexdigest()
        model_base64 = base64.b64encode(model_data).decode('utf-8')
        timestamp = datetime.datetime.now().isoformat()
        
        firebase_data = {
            'model_base64': model_base64,
            'metadata': {
                'version': timestamp,
                'hash': model_hash,
                'size': len(model_data),
                'uploaded_at': timestamp,
                'scaler_mean': scaler_mean,
                'scaler_scale': scaler_scale,
                'num_features': len(scaler_mean),
                'max_lag': max_lag,
                # NUEVA METADATA DEL MODELO
                'architecture': model_config.get('architecture', []),
                'epochs': model_config.get('epochs', 0),
                'batch_size': model_config.get('batch_size', 0),
                'learning_rate': model_config.get('learning_rate', 'adam'),
                'optimizer': model_config.get('optimizer', 'adam'),
                'loss_function': model_config.get('loss_function', 'mse'),
                'total_params': model_config.get('total_params', 0),
                'training_mode': model_config.get('mode', 'auto')
            }
        }
        
        url = f"{FIREBASE_DB_URL}/ml_model.json"
        
        if use_auth:
            token = get_firebase_token()
            if token:
                url += f"?auth={token}"
            else:
                return {'success': False, 'error': 'No se pudo obtener token'}
        
        response = requests.put(url, json=firebase_data)
        
        if response.status_code == 200:
            return {
                'success': True, 
                'hash': model_hash, 
                'version': timestamp,
                'metadata': firebase_data['metadata']
            }
        else:
            return {'success': False, 'error': response.text}
    
    except Exception as e:
        print(f"[ERROR] upload_to_firebase: {str(e)}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

# ==================== RUTAS FLASK ====================

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[ERROR] Excepción: {str(e)}")
    traceback.print_exc()
    return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        file_paths = {}
        
        for key in ['pm1', 'pm25', 'pm10']:
            if key in request.files:
                file = request.files[key]
                if file.filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{key}.csv')
                    file.save(filepath)
                    file_paths[key] = filepath
        
        if not file_paths:
            return jsonify({'error': 'No se subieron archivos'}), 400
        
        print(f"[DEBUG] Archivos subidos: {list(file_paths.keys())}")
        
        daily_df = load_and_combine_data(file_paths)
        
        if daily_df is None or len(daily_df) == 0:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 400
        
        print(f"[DEBUG] DataFrame - Shape: {daily_df.shape}, Columnas: {daily_df.columns.tolist()}")
        
        num_variables = int(daily_df.shape[1])
        
        stats = {
            'rows': int(daily_df.shape[0]),
            'columns': num_variables,
            'date_start': daily_df.index.min().strftime('%Y-%m-%d'),
            'date_end': daily_df.index.max().strftime('%Y-%m-%d'),
            'columns_list': daily_df.columns.tolist(),
            'preview': daily_df.head(10).to_dict('records')
        }
        
        print(f"[DEBUG] Stats: rows={stats['rows']}, columns={stats['columns']}")
        
        return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        print(f"[ERROR] upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    try:
        config = request.json
        print(f"[DEBUG] Config recibida: {config}")
        
        file_paths = {
            'pm1': os.path.join(app.config['UPLOAD_FOLDER'], 'pm1.csv'),
            'pm25': os.path.join(app.config['UPLOAD_FOLDER'], 'pm25.csv'),
            'pm10': os.path.join(app.config['UPLOAD_FOLDER'], 'pm10.csv')
        }
        
        existing_files = [k for k, v in file_paths.items() if os.path.exists(v)]
        if not existing_files:
            return jsonify({'error': 'No hay archivos CSV. Carga los datos primero.'}), 400
        
        result = train_model(file_paths, config, mode=config.get('mode', 'auto'))
        
        if 'error' in result:
            return jsonify(result), 400
        
        model = result['model']
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open('model_mlp.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"[DEBUG] Modelo guardado")
        
        # EXTRAER ARQUITECTURA DEL MODELO
        architecture = []
        for layer in model.layers[:-1]:  # Excluir capa de salida
            layer_config = layer.get_config()
            architecture.append({
                'neurons': layer_config.get('units', 0),
                'activation': layer_config.get('activation', 'relu')
            })
        
        response = {
            'success': True,
            'metrics': result['metrics'],
            'history': result['history'],
            'predictions': result['predictions'],
            'scaler': {
                'mean': model._scaler.mean_.tolist(),
                'scale': model._scaler.scale_.tolist()
            },
            'max_lag': config.get('max_lag', 5),
            # NUEVA INFO DE CONFIGURACIÓN
            'model_config': {
                'architecture': architecture,
                'epochs': config.get('epochs', 200),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': 'adam',
                'optimizer': 'adam',
                'loss_function': 'mse',
                'total_params': result['metrics']['n_params'],
                'mode': config.get('mode', 'auto')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] train: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/deploy', methods=['POST'])
def deploy():
    try:
        data = request.json
        
        if not os.path.exists('model_mlp.tflite'):
            return jsonify({'error': 'No hay modelo entrenado'}), 400
        
        scaler_mean = data['scaler_mean'] if isinstance(data['scaler_mean'], list) else data['scaler_mean'].tolist()
        scaler_scale = data['scaler_scale'] if isinstance(data['scaler_scale'], list) else data['scaler_scale'].tolist()
        
        max_lag = data.get('max_lag', 5)
        model_config = data.get('model_config', {})
        
        result = upload_to_firebase(
            'model_mlp.tflite', 
            scaler_mean, 
            scaler_scale, 
            max_lag,
            model_config,
            use_auth=True
        )
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] deploy: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download')
def download():
    try:
        if not os.path.exists('model_mlp.tflite'):
            return jsonify({'error': 'No hay modelo disponible'}), 404
        return send_file('model_mlp.tflite', as_attachment=True)
    except Exception as e:
        print(f"[ERROR] download: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 404


@app.route('/backtest', methods=['POST'])
def backtest():
    try:
        config = request.json
        print(f"[DEBUG] Config backtest: {config}")

        file_paths = {
            'pm1': os.path.join(app.config['UPLOAD_FOLDER'], 'pm1.csv'),
            'pm25': os.path.join(app.config['UPLOAD_FOLDER'], 'pm25.csv'),
            'pm10': os.path.join(app.config['UPLOAD_FOLDER'], 'pm10.csv')
        }

        existing_files = [k for k, v in file_paths.items() if os.path.exists(v)]
        if not existing_files:
            return jsonify({'error': 'No hay archivos CSV. Carga los datos primero.'}), 400

        # Entrenar modelo para validación
        result = train_model(file_paths, config, mode='auto')
        if 'error' in result:
            return jsonify(result), 400

        # Extraer predicciones
        y_test = np.array(result['predictions']['y_test'])
        y_pred = np.array(result['predictions']['y_pred'])
        dates = result['predictions']['dates']

        # Calcular métricas detalladas
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100)
        max_error = float(np.max(np.abs(y_test - y_pred)))

        # Función para categorizar AQI
        def get_aqi_category(pm25):
            if pm25 <= 12:
                return "Good"
            elif pm25 <= 35.4:
                return "Moderate"
            elif pm25 <= 55.4:
                return "Unhealthy for Sensitive Groups"
            elif pm25 <= 150.4:
                return "Unhealthy"
            elif pm25 <= 250.4:
                return "Very Unhealthy"
            else:
                return "Hazardous"

        # Calcular precisión de categorías
        real_categories = [get_aqi_category(val) for val in y_test]
        pred_categories = [get_aqi_category(val) for val in y_pred]
        category_accuracy = float(np.mean([r == p for r, p in zip(real_categories, pred_categories)]) * 100)

        # Resultados diarios
        daily_results = []
        for i, date in enumerate(dates):
            real = float(y_test[i])
            predicted = float(y_pred[i])
            error = abs(real - predicted)
            error_pct = (error / real) * 100 if real != 0 else 0
            real_cat = real_categories[i]
            correct = real_cat == pred_categories[i]

            daily_results.append({
                'date': date,
                'real': real,
                'predicted': predicted,
                'error': error,
                'error_pct': error_pct,
                'real_category': real_cat,
                'correct_category': correct
            })

        # Evaluación general
        if mae < 10 and r2 > 0.7 and category_accuracy > 70:
            quality = 'excellent'
            message = '🌟 Excelente rendimiento del modelo. Listo para producción.'
            rating = 5
        elif mae < 15 and r2 > 0.5 and category_accuracy > 60:
            quality = 'good'
            message = '✅ Buen rendimiento. Recomendado para uso operativo.'
            rating = 4
        else:
            quality = 'moderate'
            message = '⚠️ Rendimiento aceptable. Considera ajustes adicionales.'
            rating = 3

        recommendations = []
        if mae > 15:
            recommendations.append("Considera aumentar el número de epochs o ajustar la arquitectura.")
        if r2 < 0.5:
            recommendations.append("El modelo tiene bajo ajuste. Revisa las características de entrada.")
        if category_accuracy < 60:
            recommendations.append("La precisión de categorías AQI es baja. Mejora la predicción de niveles.")
        if not recommendations:
            recommendations.append("El modelo muestra buen rendimiento general.")

        evaluation = {
            'quality': quality,
            'message': message,
            'rating': rating,
            'recommendations': recommendations
        }

        backtest_data = {
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'max_error': max_error,
                'category_accuracy': category_accuracy
            },
            'evaluation': evaluation,
            'daily_results': daily_results
        }

        return jsonify({'success': True, 'backtest': backtest_data})

    except Exception as e:
        print(f"[ERROR] backtest: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    print("=" * 60)
    print("🚀 Servidor Flask iniciando...")
    print("📍 URL: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True,
        request_handler=WSGIRequestHandler
    )