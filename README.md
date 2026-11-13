# Dashboard · Decisión de Crédito (Fintech Vehicular)

Dashboard interactivo analítico y predictivo desarrollado con **Plotly Dash** para evaluación y otorgamiento de préstamos vehiculares. Preparado para despliegue en **Render** (plan gratuito: 512 MB RAM, entorno Linux) en la rama `staging`.

---

## 1. Arquitectura y Archivos en Producción / Staging

- **`app.py`**: Aplicación Dash + servidor WSGI Flask (`server = app.server`). Carga `modelo_rf.pkl` y `dataset_limpio.csv` usando rutas relativas seguras (`BASE_DIR = Path(__file__).resolve().parent`).
- **`modelo_rf.pkl`**: Pipeline de Machine Learning (Random Forest) serializado con `scikit-learn==1.6.1`.
- **`dataset_limpio.csv`**: Cartera histórica procesada (99,810 registros) para KPIs dinámicos y análisis de riesgo.
- **`requirements.txt`**: Dependencias estrictamente necesarias para el runtime en producción.
- **`runtime.txt`**: Versión fijada de Python (`python-3.11.11`) para compatibilidad garantizada en Linux.
- **`Procfile`**: Definición del proceso web para Render con Gunicorn.

---

## 2. Ejecución en Local

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Instalar dependencias optimizadas
pip install -r requirements.txt

# 3. Iniciar el servidor
python app.py
```

El dashboard estará disponible en `http://127.0.0.1:8050/`.

---

## 3. Configuración para Despliegue en Render (`staging`)

1. En [Render Dashboard](https://dashboard.render.com), haz clic en **New +** > **Web Service**.
2. Conecta tu repositorio de GitHub (`DASHBOARD`).
3. En la configuración del servicio:
   - **Name**: `dashboard-fintech-staging` (o el nombre preferido).
   - **Branch**: `staging`
   - **Root Directory**: *(dejar en blanco / raíz)*
   - **Runtime**: `Python` (Render detectará automáticamente `runtime.txt` con Python 3.11).
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
     ```
     *(Alternativa directa si no se usa Gunicorn: `python app.py`)*
   - **Instance Type**: `Free` (0.1 CPU, 512 MB RAM).

> **Nota MLOps (Gestión de Memoria 512 MB):**  
> Se configura exactamente **1 worker** con **2 threads** en Gunicorn. Cada worker adicional duplica en memoria el dataset y el modelo (`~140 MB` por proceso). Mantener `1 worker` asegura un consumo estable entre 130 y 160 MB de RAM, previniendo reinicios por Out-Of-Memory (OOM) en el plan gratuito.
