// ===== Sistema de Entrenamiento PM2.5 - Machine Learning =====

// Variables globales
let modelData = null;
let charts = {};

// ===== Inicialización =====
document.addEventListener('DOMContentLoaded', () => {
  initializeTabs();
  initializeSliders();
  initializeModeToggle();
  initializeFileInputs();
  initializeButtons();
  
  console.log('Sistema de entrenamiento inicializado correctamente');
});

// ===== Gestión de Tabs =====
function initializeTabs() {
  const tabs = document.querySelectorAll('.tab');
  const tabContents = document.querySelectorAll('.tab-content');
  
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.dataset.tab;
      
      // Remover active de todos
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      // Activar seleccionado
      tab.classList.add('active');
      document.getElementById(`tab-${targetTab}`).classList.add('active');
    });
  });
}

// ===== Gestión de Sliders =====
function initializeSliders() {
  const sliders = [
    { id: 'tuning-time', valueId: 'tuning-time-value', suffix: '' },
    { id: 'epochs', valueId: 'epochs-value', suffix: '' },
    { id: 'test-size', valueId: 'test-size-value', multiplier: 100, suffix: '' },
    { id: 'max-lag', valueId: 'max-lag-value', suffix: '' },
    { id: 'num-layers', valueId: 'num-layers-value', suffix: '' },
    { id: 'backtest-days', valueId: 'backtest-days-value', suffix: '' }
  ];
  
  sliders.forEach(({ id, valueId, multiplier, suffix }) => {
    const slider = document.getElementById(id);
    const valueSpan = document.getElementById(valueId);
    
    if (!slider || !valueSpan) return;
    
    slider.addEventListener('input', () => {
      const value = multiplier ? slider.value * multiplier : slider.value;
      valueSpan.textContent = Math.round(value);
    });
  });
  
  // Listener especial para num-layers
  const numLayersSlider = document.getElementById('num-layers');
  if (numLayersSlider) {
    numLayersSlider.addEventListener('input', updateLayersUI);
  }
}

// ===== Gestión de Modo (Auto/Manual) =====
function initializeModeToggle() {
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  
  modeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      const isManual = document.getElementById('mode-manual').checked;
      
      document.getElementById('auto-config').classList.toggle('hidden', isManual);
      document.getElementById('manual-layers').classList.toggle('hidden', !isManual);
      
      // ✅ NUEVO: Mostrar/ocultar selector de método
      const tuningMethodDiv = document.getElementById('auto-tuning-method');
      if (tuningMethodDiv) {
        tuningMethodDiv.classList.toggle('hidden', isManual);
      }
      
      if (isManual) {
        updateLayersUI();
      }
    });
  });
  
  // ✅ NUEVO: Inicializar selector de método de tuning
  initializeTuningMethodSelector();
}

// ✅ NUEVA FUNCIÓN: Manejar cambios en el selector de método
function initializeTuningMethodSelector() {
  const tuningMethodSelect = document.getElementById('tuning-method');
  const methodDescription = document.getElementById('method-description');
  const methodDetails = document.getElementById('method-details');
  
  if (!tuningMethodSelect || !methodDescription || !methodDetails) return;
  
  const methodInfo = {
    'pca_kmeans': {
      title: 'PCA + K-Means:',
      description: 'Determina capas con análisis de componentes principales y neuronas con clustering. Basado en investigación científica publicada.'
    },
    'grid_search': {
      title: 'Grid Search:',
      description: 'Prueba múltiples combinaciones de arquitecturas y selecciona la mejor. Más lento pero exhaustivo.'
    },
    'hybrid': {
      title: 'Híbrido:',
      description: 'Compara ambos métodos (PCA+K-Means vs Grid Search) y selecciona el de mejor rendimiento. Tiempo de ejecución duplicado.'
    }
  };
  
  tuningMethodSelect.addEventListener('change', (e) => {
    const method = e.target.value;
    const info = methodInfo[method];
    
    if (info) {
      methodDescription.textContent = info.title;
      methodDetails.textContent = info.description;
    }
  });
}

// ===== Actualizar UI de Capas Manuales =====
function updateLayersUI() {
  const numLayers = parseInt(document.getElementById('num-layers').value);
  const container = document.getElementById('layers-container');
  
  if (!container) return;
  
  container.innerHTML = '';
  
  for (let i = 0; i < numLayers; i++) {
    const layerDiv = document.createElement('div');
    layerDiv.className = 'layer-config';
    layerDiv.innerHTML = `
      <div class="layer-header">Capa ${i + 1}</div>
      
      <div class="slider-group">
        <label>Neuronas:</label>
        <input type="range" class="layer-neurons" min="4" max="64" value="16" data-layer="${i}">
        <div class="slider-value">
          <span class="neurons-value-${i}">16</span>
          <span class="slider-unit">neuronas</span>
        </div>
      </div>
      
      <div class="select-group">
        <label>Activación:</label>
        <select class="layer-activation" data-layer="${i}">
          <option value="relu" selected>ReLU</option>
          <option value="tanh">Tanh</option>
          <option value="sigmoid">Sigmoid</option>
          <option value="elu">ELU</option>
        </select>
      </div>
    `;
    
    container.appendChild(layerDiv);
    
    // Agregar listener al slider de neuronas
    const neuronSlider = layerDiv.querySelector('.layer-neurons');
    neuronSlider.addEventListener('input', (e) => {
      document.querySelector(`.neurons-value-${i}`).textContent = e.target.value;
    });
  }
}

// ===== Gestión de File Inputs =====
function initializeFileInputs() {
  const fileInputs = ['pm1', 'pm25', 'pm10'];
  
  fileInputs.forEach(type => {
    const input = document.getElementById(`file-${type}`);
    const filenameSpan = document.getElementById(`filename-${type}`);
    
    if (!input || !filenameSpan) return;
    
    // Hacer que el span active el input
    filenameSpan.addEventListener('click', () => input.click());
    
    // Actualizar nombre cuando se selecciona archivo
    input.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        filenameSpan.textContent = e.target.files[0].name;
        filenameSpan.style.color = 'var(--text-accent)';
      } else {
        filenameSpan.textContent = 'Ningún archivo seleccionado';
        filenameSpan.style.color = 'var(--text-muted)';
      }
    });
  });
}

// ===== Gestión de Botones =====
function initializeButtons() {
  const uploadBtn = document.getElementById('btn-upload');
  const trainBtn = document.getElementById('btn-train');
  const backtestBtn = document.getElementById('btn-backtest');

  if (backtestBtn) {
    backtestBtn.addEventListener('click', handleBacktest);
  }
  
  if (uploadBtn) {
    uploadBtn.addEventListener('click', handleUpload);
  }
  
  if (trainBtn) {
    trainBtn.addEventListener('click', handleTrain);
  }
}

// ===== Manejo de Upload =====
async function handleUpload() {
  const formData = new FormData();
  const files = ['pm1', 'pm25', 'pm10'];
  let hasFiles = false;
  
  files.forEach(type => {
    const fileInput = document.getElementById(`file-${type}`);
    if (fileInput && fileInput.files.length > 0) {
      formData.append(type, fileInput.files[0]);
      hasFiles = true;
    }
  });
  
  if (!hasFiles) {
    showAlert('tab-datos', 'error', 'Debes subir al menos un archivo CSV');
    return;
  }
  
  showLoading('Cargando datos...', 'Procesando archivos CSV');
  
  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
      displayDataInfo(data.stats);
      document.getElementById('btn-train').disabled = false;
      document.getElementById('btn-backtest').disabled = false;
      //showAlert('tab-datos', 'success', '✅ Datos cargados exitosamente');
    } else {
      showAlert('tab-datos', 'error', data.error || 'Error cargando datos');
    }
  } catch (error) {
    console.error('Error en upload:', error);
    showAlert('tab-datos', 'error', 'Error de conexión: ' + error.message);
  } finally {
    hideLoading();
  }
}

// ===== Manejo de Entrenamiento =====
async function handleTrain() {
  const mode = document.getElementById('mode-manual').checked ? 'manual' : 'auto';
  
  const config = {
    mode: mode,
    epochs: parseInt(document.getElementById('epochs').value),
    batch_size: parseInt(document.getElementById('batch-size').value),
    test_size: parseFloat(document.getElementById('test-size').value),
    max_lag: parseInt(document.getElementById('max-lag').value)
  };
  
  // Configuración específica según modo
  if (mode === 'auto') {
    config.tuning_time = parseInt(document.getElementById('tuning-time').value);
    const tuningMethodSelect = document.getElementById('tuning-method');
    if (tuningMethodSelect) {
      config.tuning_method = tuningMethodSelect.value;
    } else {
      config.tuning_method = 'pca_kmeans';  // Default si no existe el selector
    }
  } else {
    const numLayers = parseInt(document.getElementById('num-layers').value);
    config.layers = [];
    
    for (let i = 0; i < numLayers; i++) {
      const neurons = parseInt(document.querySelector(`.layer-neurons[data-layer="${i}"]`).value);
      const activation = document.querySelector(`.layer-activation[data-layer="${i}"]`).value;
      config.layers.push({ neurons, activation });
    }
  }
  
  showLoading('Entrenando modelo...', 'Este proceso puede tomar varios minutos');
  document.getElementById('btn-train').disabled = true;
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1800000); // 30 minutos
    
    const response = await fetch('/train', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Connection': 'keep-alive'
      },
      body: JSON.stringify(config),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    const data = await response.json();
    
    if (data.success) {
      modelData = data;
      modelData.tuning_method = config.tuning_method || 'auto';
      displayTrainingResults(data);
      displayPredictionResults(data);
      enableDeployment(data);
      
      // Cambiar a tab de resultados
      document.querySelector('.tab[data-tab="resultados"]').click();
    } else {
      showAlert('tab-entrenamiento', 'error', data.error || 'Error en el entrenamiento');
    }
  } catch (error) {
    console.error('Error en entrenamiento:', error);
    if (error.name === 'AbortError') {
      showAlert('tab-entrenamiento', 'error', '⏱️ Timeout: El entrenamiento excedió 30 minutos');
    } else {
      showAlert('tab-entrenamiento', 'error', '❌ Error de conexión: ' + error.message);
    }
  } finally {
    hideLoading();
    document.getElementById('btn-train').disabled = false;
  }
}

// ===== Mostrar Información de Datos =====
function displayDataInfo(stats) {
  const html = `
    <div class="alert alert-success">
      ✅ Datos cargados exitosamente
    </div>
    
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">${stats.rows}</div>
        <div class="stat-label">Días de datos</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.columns}</div>
        <div class="stat-label">Variables</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.date_start}</div>
        <div class="stat-label">Fecha inicio</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.date_end}</div>
        <div class="stat-label">Fecha fin</div>
      </div>
    </div>
    
    <h3 style="color: var(--text-accent); margin-bottom: var(--spacing-md);">Vista previa de datos</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            ${stats.columns_list.map(col => `<th>${col}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          ${stats.preview.map(row => `
            <tr>
              ${stats.columns_list.map(col => `<td>${row[col]?.toFixed(2) || 'N/A'}</td>`).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
  
  document.getElementById('tab-datos').innerHTML = html;
}

// ===== Mostrar Resultados de Entrenamiento =====
function displayTrainingResults(data) {
  const html = `
    <div class="alert alert-success">
      ✅ Modelo entrenado exitosamente
    </div>
    
    <h3 style="color: var(--text-accent); margin-bottom: var(--spacing-md);">Métricas del Modelo</h3>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${data.metrics.test_mse.toFixed(4)}</div>
        <div class="metric-label">MSE Validación</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.test_mae.toFixed(4)}</div>
        <div class="metric-label">MAE Validación</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.test_r2.toFixed(4)}</div>
        <div class="metric-label">R² Validación</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.train_mse.toFixed(4)}</div>
        <div class="metric-label">MSE Entrenamiento</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.train_mae.toFixed(4)}</div>
        <div class="metric-label">MAE Entrenamiento</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.n_params}</div>
        <div class="metric-label">Parámetros</div>
      </div>
    </div>
    
    <div class="chart-container">
      <h3>Curva de Entrenamiento</h3>
      <canvas id="training-chart"></canvas>
    </div>
  `;
  
  document.getElementById('tab-entrenamiento').innerHTML = html;
  
  // Crear gráfica de pérdida
  const ctx = document.getElementById('training-chart').getContext('2d');
  
  if (charts.training) {
    charts.training.destroy();
  }
  
  charts.training = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.history.loss.map((_, i) => i + 1),
      datasets: [
        {
          label: 'Train Loss',
          data: data.history.loss,
          borderColor: '#ffbb00ff',
          backgroundColor: 'rgba(0, 212, 255, 0.1)',
          borderWidth: 2,
          tension: 0.4,
          fill: true
        },
        {
          label: 'Validation Loss',
          data: data.history.val_loss,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          borderWidth: 2,
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#e6edf3', font: { family: 'Inter' } }
        }
      },
      scales: {
        x: {
          ticks: { color: '#a2a9b1' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#a2a9b1' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// ===== Mostrar Resultados de Predicción =====
function displayPredictionResults(data) {
  const html = `
    <div class="alert alert-success">
      📊 Resultados del modelo
    </div>
    
    <div class="chart-container">
      <h3>Predicción vs Real</h3>
      <canvas id="scatter-chart"></canvas>
    </div>
    
    <div class="chart-container">
      <h3>Serie Temporal</h3>
      <canvas id="timeseries-chart"></canvas>
    </div>
  `;
  
  document.getElementById('tab-resultados').innerHTML = html;
  
  // Scatter plot
  const scatterCtx = document.getElementById('scatter-chart').getContext('2d');
  
  if (charts.scatter) {
    charts.scatter.destroy();
  }
  
  charts.scatter = new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Predicciones',
        data: data.predictions.y_test.map((real, i) => ({
          x: real,
          y: data.predictions.y_pred[i]
        })),
        backgroundColor: 'rgba(0, 212, 255, 0.6)',
        borderColor: '#00e5ffff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'PM2.5 Real (μg/m³)', color: '#a2a9b1' },
          ticks: { color: '#a2a9b1' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          title: { display: true, text: 'PM2.5 Predicho (μg/m³)', color: '#a2a9b1' },
          ticks: { color: '#a2a9b1' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
  
  // Time series
  const timeseriesCtx = document.getElementById('timeseries-chart').getContext('2d');
  
  if (charts.timeseries) {
    charts.timeseries.destroy();
  }
  
  charts.timeseries = new Chart(timeseriesCtx, {
    type: 'line',
    data: {
      labels: data.predictions.dates,
      datasets: [
        {
          label: 'Real',
          data: data.predictions.y_test,
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(246, 255, 0, 0.1)',
          borderWidth: 2,
          tension: 0.4,
          fill: true
        },
        {
          label: 'Predicción',
          data: data.predictions.y_pred,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          borderWidth: 2,
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#e6edf3', font: { family: 'Inter' } }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Fecha', color: '#a2a9b1' },
          ticks: { color: '#a2a9b1', maxRotation: 45, minRotation: 45 },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          title: { display: true, text: 'PM2.5 (μg/m³)', color: '#a2a9b1' },
          ticks: { color: '#a2a9b1' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// ===== Habilitar Despliegue =====
function enableDeployment(data) {
  const meanArray = data.scaler.mean.map(v => `  ${v.toFixed(6)}`).join(',\n');
  const scaleArray = data.scaler.scale.map(v => `  ${v.toFixed(6)}`).join(',\n');
  
  const html = `
    <div class="alert alert-success">
      ✅ Modelo listo para desplegar
    </div>
    
    <h3 style="color: var(--text-accent); margin-bottom: var(--spacing-md);">Información del Modelo</h3>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${data.metrics.n_params}</div>
        <div class="metric-label">Parámetros</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.scaler.mean.length}</div>
        <div class="metric-label">Features</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${data.metrics.test_r2.toFixed(3)}</div>
        <div class="metric-label">R² Score</div>
      </div>
    </div>
    
    <h3 style="color: var(--text-accent); margin: var(--spacing-xl) 0 var(--spacing-md) 0;">
      Parámetros de Normalización (para ESP32)
    </h3>
    <div class="code-block">
<span style="color: #c678dd;">// Scaler Mean</span>
<span style="color: #e06c75;">const float</span> input_mean[] = {
${meanArray}
};

<span style="color: #c678dd;">// Scaler Scale</span>
<span style="color: #e06c75;">const float</span> input_scale[] = {
${scaleArray}
};

<span style="color: #e06c75;">const int</span> num_features = <span style="color: #d19a66;">${data.scaler.mean.length}</span>;
    </div>
    
    <h3 style="color: var(--text-accent); margin: var(--spacing-xl) 0 var(--spacing-md) 0;">
      Opciones de Despliegue
    </h3>
    <div class="deployment-buttons">
      <button class="btn btn-download" id="btn-download">
        <span class="btn-icon">⬇️</span>
        Descargar model_mlp.tflite
      </button>
      <button class="btn btn-firebase" id="btn-firebase">
        <span class="btn-icon">☁️</span>
        Subir a Firebase
      </button>
    </div>
    
    <div id="firebase-result"></div>
  `;
  
  document.getElementById('tab-despliegue').innerHTML = html;
  
  // Download button
  document.getElementById('btn-download').addEventListener('click', () => {
    window.location.href = '/download';
  });
  
  // Firebase button
  document.getElementById('btn-firebase').addEventListener('click', handleFirebaseUpload);
}

// ===== Manejo de Upload a Firebase =====
async function handleFirebaseUpload() {
  if (!modelData) {
    showAlert('firebase-result', 'error', 'No hay datos del modelo disponibles');
    return;
  }
  
  const btn = document.getElementById('btn-firebase');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Subiendo...';
  const modelConfigWithMethod = {
    ...modelData.model_config,
    tuning_method: modelData.tuning_method || 'auto'  // Preservar método usado
  };

  try {
    const response = await fetch('/deploy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scaler_mean: modelData.scaler.mean,
        scaler_scale: modelData.scaler.scale,
        max_lag: modelData.max_lag,
        model_config: modelConfigWithMethod  // ✅ ACTUALIZADO
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      displayFirebaseDeployment(result);
    } else {
      document.getElementById('firebase-result').innerHTML = `
        <div class="alert alert-error">
          ❌ Error: ${result.error}
        </div>
      `;
    }
  } catch (error) {
    console.error('Error en upload Firebase:', error);
    document.getElementById('firebase-result').innerHTML = `
      <div class="alert alert-error">
        ❌ Error de conexión: ${error.message}
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">☁️</span> Subir a Firebase';
  }
}

// ===== Nueva función para mostrar detalles del deployment =====
function displayFirebaseDeployment(result) {
  const metadata = result.metadata;
  const architecture = metadata.architecture || [];
  const tuningMethod = metadata.tuning_method || 'auto';
  const methodNames = {
    'pca_kmeans': '🎓 PCA + K-Means (Paper)',
    'grid_search': '🔍 Grid Search (Exhaustivo)',
    'hybrid': '⚖️ Híbrido (Comparativo)',
    'auto': '⚡ Auto-Tuning (Genérico)'
  };

  const methodName = methodNames[tuningMethod] || '⚡ Auto-Tuning';
  
  // Generar tabla de capas
  const layersTable = architecture.map((layer, idx) => `
    <tr>
      <td>Capa ${idx + 1}</td>
      <td>${layer.neurons}</td>
      <td>${layer.activation.toUpperCase()}</td>
    </tr>
  `).join('');
  
  const html = `
    <div class="firebase-deployment-info">
      <div class="alert alert-success">
        ✅ Modelo desplegado exitosamente en Firebase
      </div>
      
      <div class="deployment-section">
        <h4>📊 Información del Despliegue</h4>
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">Hash del Modelo:</span>
            <code>${result.hash}</code>
          </div>
          <div class="info-item">
            <span class="info-label">Versión:</span>
            <span>${new Date(result.version).toLocaleString('es-ES')}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Tamaño:</span>
            <span>${(metadata.size / 1024).toFixed(2)} KB</span>
          </div>
        </div>
      </div>
      
      <div class="deployment-section">
        <h4>🧠 Arquitectura de la Red Neuronal</h4>
        <div class="architecture-grid">
          <div class="arch-card">
            <div class="arch-value">${architecture.length}</div>
            <div class="arch-label">Capas Ocultas</div>
          </div>
          <div class="arch-card">
            <div class="arch-value">${metadata.total_params}</div>
            <div class="arch-label">Parámetros Totales</div>
          </div>
          <div class="arch-card">
            <div class="arch-value">${metadata.num_features}</div>
            <div class="arch-label">Features de Entrada</div>
          </div>
          <div class="arch-card">
            <div class="arch-value">${metadata.max_lag}</div>
            <div class="arch-label">Lags Temporales</div>
          </div>
        </div>
        
        <h5 style="margin-top: 1.5rem; color: var(--text-accent);">Configuración de Capas</h5>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Capa</th>
                <th>Neuronas</th>
                <th>Activación</th>
              </tr>
            </thead>
            <tbody>
              ${layersTable}
              <tr style="background: rgba(88, 166, 255, 0.1);">
                <td><strong>Salida</strong></td>
                <td><strong>1</strong></td>
                <td><strong>LINEAR</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <div class="deployment-section">
        <h4>⚙️ Hiperparámetros de Entrenamiento</h4>
        <div class="params-grid">
          <div class="param-item">
            <span class="param-label">Modo de Entrenamiento:</span>
            <span class="param-value">${metadata.training_mode === 'auto' ? '⚡ Auto-Tuning' : '🎛️ Manual'}</span>
          </div>

          <!-- ✅ NUEVO: Mostrar método de auto-tuning usado -->
          ${metadata.training_mode === 'auto' ? `
          <div class="param-item">
            <span class="param-label">Método de Optimización:</span>
            <span class="param-value">${methodName}</span>
          </div>
          ` : ''}

          <div class="param-item">
            <span class="param-label">Épocas:</span>
            <span class="param-value">${metadata.epochs}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Batch Size:</span>
            <span class="param-value">${metadata.batch_size}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Optimizador:</span>
            <span class="param-value">${metadata.optimizer.toUpperCase()}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Función de Pérdida:</span>
            <span class="param-value">${metadata.loss_function.toUpperCase()}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Learning Rate:</span>
            <span class="param-value">Adam (default)</span>
          </div>
        </div>
      </div>
      
      <div class="deployment-section">
        <h4>🎯 Estado del Despliegue</h4>
        <div class="status-box">
          <div class="status-icon">✅</div>
          <div class="status-text">
            <strong>Modelo Activo en Firebase</strong>
            <p>El ESP32 puede descargar y ejecutar este modelo automáticamente.</p>
            <p>URL: <code>https://esp32-pms7003-database-system-default-rtdb.firebaseio.com/ml_model.json</code></p>
          </div>
        </div>
      </div>
    </div>
  `;
  
  document.getElementById('firebase-result').innerHTML = html;
}

// ===== Utilidades: Loading Panel (no intrusivo) =====
let loadingStartTime = null;
let loadingInterval = null;

function showLoading(title, message) {
  const panel = document.getElementById('loadingPanel');
  const titleEl = document.getElementById('loadingTitle');
  const messageEl = document.getElementById('loadingMessage');
  const progressBar = document.getElementById('progressBar');
  const timeEl = document.getElementById('loadingTime');
  
  if (panel && titleEl && messageEl) {
    titleEl.textContent = title;
    messageEl.textContent = message;
    panel.classList.remove('hidden');
    
    // Iniciar contador de tiempo
    loadingStartTime = Date.now();
    progressBar.style.width = '0%';
    
    // Simular progreso
    let progress = 0;
    if (loadingInterval) clearInterval(loadingInterval);
    
    loadingInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - loadingStartTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      
      timeEl.textContent = `Tiempo transcurrido: ${minutes}:${seconds.toString().padStart(2, '0')}`;
      
      // Progreso simulado (asintótico)
      if (progress < 90) {
        progress += Math.random() * 2;
        progressBar.style.width = `${Math.min(progress, 90)}%`;
      }
    }, 1000);
  }
}

function hideLoading() {
  const panel = document.getElementById('loadingPanel');
  const progressBar = document.getElementById('progressBar');
  
  if (panel) {
    // Completar barra
    progressBar.style.width = '100%';
    
    setTimeout(() => {
      panel.classList.add('hidden');
      if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
      }
      loadingStartTime = null;
    }, 500);
  }
}

// ===== Utilidades: Alertas =====
function showAlert(containerId, type, message) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  const alertDiv = document.createElement('div');
  alertDiv.className = `alert alert-${type}`;
  alertDiv.textContent = message;
  
  // Insertar al inicio del contenedor
  container.insertBefore(alertDiv, container.firstChild);
  
  // Auto-remover después de 5 segundos para errores
  if (type === 'error') {
    setTimeout(() => {
      alertDiv.style.opacity = '0';
      setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
  }
}

// ===== Manejo de Errores Globales =====
window.addEventListener('error', (error) => {
  console.error('Error global:', error);
  hideLoading();
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Promise rechazado:', event.reason);
  hideLoading();
});

// ===== BACKTESTING =====
async function handleBacktest() {
  const test_days = parseInt(document.getElementById('backtest-days').value);
  
  const config = {
    test_days: test_days,
    max_lag: parseInt(document.getElementById('max-lag').value),
    epochs: 100,
    test_size: 0.2
  };
  
  showLoading('Ejecutando validación...', 'Evaluando modelo con datos históricos');
  document.getElementById('btn-backtest').disabled = true;
  
  try {
    const response = await fetch('/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    const data = await response.json();
    
    if (data.success) {
      displayBacktestResults(data.backtest);
      document.querySelector('.tab[data-tab="validacion"]').click();
    } else {
      showAlert('tab-validacion', 'error', data.error || 'Error en validación');
    }
  } catch (error) {
    console.error('Error:', error);
    showAlert('tab-validacion', 'error', 'Error de conexión: ' + error.message);
  } finally {
    hideLoading();
    document.getElementById('btn-backtest').disabled = false;
  }
}

function displayBacktestResults(backtest) {
  const metrics = backtest.metrics;
  const evaluation = backtest.evaluation;
  const dailyResults = backtest.daily_results;
  
  const stars = '⭐'.repeat(evaluation.rating) + '☆'.repeat(5 - evaluation.rating);
  
  const dailyRows = dailyResults.map(day => `
    <tr class="${day.correct_category ? '' : 'error-row'}">
      <td>${day.date}</td>
      <td>${day.real.toFixed(2)}</td>
      <td>${day.predicted.toFixed(2)}</td>
      <td style="color: ${day.error < 10 ? 'var(--status-excellent)' : day.error < 15 ? 'var(--status-moderate)' : 'var(--status-critical)'}">
        ${day.error.toFixed(2)}
      </td>
      <td>${day.error_pct.toFixed(1)}%</td>
      <td>${day.real_category}</td>
      <td>${day.correct_category ? '✅' : '❌'}</td>
    </tr>
  `).join('');
  
  const html = `
    <div class="alert alert-${evaluation.quality === 'excellent' || evaluation.quality === 'good' ? 'success' : 'warning'}">
      ${evaluation.message}
    </div>
    
    <div class="model-rating">
      <h3>📊 Evaluación del Modelo</h3>
      <div class="rating-stars">${stars}</div>
      <div class="rating-score">${evaluation.rating}/5</div>
    </div>
    
    <h3 style="color: var(--text-accent); margin: var(--spacing-xl) 0 var(--spacing-md) 0;">
      📈 Métricas de Rendimiento
    </h3>
    <div class="metrics-grid">
      <div class="metric-card ${metrics.mae < 10 ? 'metric-excellent' : metrics.mae < 15 ? 'metric-good' : 'metric-poor'}">
        <div class="metric-value">${metrics.mae.toFixed(2)}</div>
        <div class="metric-label">MAE (μg/m³)</div>
        <div class="metric-status">${metrics.mae < 10 ? '🌟 Excelente' : metrics.mae < 15 ? '✅ Bueno' : '⚠️ Regular'}</div>
      </div>
      
      <div class="metric-card">
        <div class="metric-value">${metrics.rmse.toFixed(2)}</div>
        <div class="metric-label">RMSE (μg/m³)</div>
      </div>
      
      <div class="metric-card ${metrics.r2 > 0.7 ? 'metric-excellent' : metrics.r2 > 0.5 ? 'metric-good' : 'metric-poor'}">
        <div class="metric-value">${metrics.r2.toFixed(3)}</div>
        <div class="metric-label">R² Score</div>
        <div class="metric-status">${metrics.r2 > 0.7 ? '🌟 Excelente' : metrics.r2 > 0.5 ? '✅ Bueno' : '⚠️ Regular'}</div>
      </div>
      
      <div class="metric-card">
        <div class="metric-value">${metrics.mape.toFixed(1)}%</div>
        <div class="metric-label">MAPE</div>
      </div>
      
      <div class="metric-card">
        <div class="metric-value">${metrics.max_error.toFixed(2)}</div>
        <div class="metric-label">Error Máximo</div>
      </div>
      
      <div class="metric-card ${metrics.category_accuracy > 70 ? 'metric-excellent' : metrics.category_accuracy > 60 ? 'metric-good' : 'metric-poor'}">
        <div class="metric-value">${metrics.category_accuracy.toFixed(1)}%</div>
        <div class="metric-label">Precisión AQI</div>
        <div class="metric-status">${metrics.category_accuracy > 70 ? '🌟 Excelente' : metrics.category_accuracy > 60 ? '✅ Bueno' : '⚠️ Regular'}</div>
      </div>
    </div>
    
    <h3 style="color: var(--text-accent); margin: var(--spacing-xl) 0 var(--spacing-md) 0;">
      📋 Resultados Detallados
    </h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Fecha</th>
            <th>Real (μg/m³)</th>
            <th>Predicho (μg/m³)</th>
            <th>Error</th>
            <th>Error %</th>
            <th>Categoría</th>
            <th>Acierto</th>
          </tr>
        </thead>
        <tbody>
          ${dailyRows}
        </tbody>
      </table>
    </div>
    
    <div class="recommendations-box">
      <h3>💡 Recomendaciones</h3>
      <ul>
        ${evaluation.recommendations.map(rec => `<li>${rec}</li>`).join('')}
      </ul>
    </div>
  `;
  
  document.getElementById('tab-validacion').innerHTML = html;
}

// ===== Cleanup al salir =====
window.addEventListener('beforeunload', () => {
  // Destruir gráficas
  Object.values(charts).forEach(chart => {
    if (chart && typeof chart.destroy === 'function') {
      try {
        chart.destroy();
      } catch (e) {
        console.error('Error destruyendo gráfica:', e);
      }
    }
  });
});