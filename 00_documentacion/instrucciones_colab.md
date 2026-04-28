# 📔 Instrucciones para Google Colab: Fine-Tuning de MarIA (Fase 5)
Este documento explica cómo llevar el entrenamiento de tu tesis a la nube para aprovechar una GPU gratuita.

---

### Paso 1: Preparación en tu PC
1.  Busca el archivo `tesis_cubanismos_colab.zip` que acabo de crear en tu carpeta `d:\Tesis`.
2.  Ten a mano el archivo `train_transformer.py`.

### Paso 2: Configuración en Google Colab
1.  Entra en [Google Colab](https://colab.research.google.com/).
2.  Crea un nuevo cuaderno (Notebook).
3.  **IMPORTANTE:** Ve a `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` y selecciona **T4 GPU** (o cualquier GPU disponible).

### Paso 3: Subida de Archivos
1.  Haz clic en el icono de la carpeta (archivos) a la izquierda de Colab.
2.  Arrastra y suelta allí el archivo `tesis_cubanismos_colab.zip` y `train_transformer.py`.

### Paso 4: Ejecución (Copia estas celdas en Colab)

#### **Celda 1: Instalación de Dependencias**
```python
!pip install transformers datasets evaluate accelerate scikit-learn
# Descomprime el dataset (asegúrate de haber subido tesis_cubanismos_colab.zip)
!unzip -o tesis_cubanismos_colab.zip
```

#### **Celda 2: Lanzar el Entrenamiento Multi-Modelo (MarIA, BERTIN, BETO)**
```python
# Este script entrenará los 3 modelos secuencialmente y generará un CSV comparativo
!python train_multi.py
```

### Paso 5: Recuperar Resultados
Una vez termine (tardará unos 15-20 min por modelo), aparecerá un archivo llamado `comparativa_final_transformers.csv` con los resultados de los 3 modelos.
1. Descarga el CSV para tu tesis.
2. También se crearán carpetas `resultados_maria`, `resultados_bertin` y `resultados_beto` con los modelos guardados.

---
**Tip Metodológico:** En tu tesis, especifica que el entrenamiento se realizó en un entorno NVIDIA Tesla T4 con 16GB de VRAM, lo que permitió procesar el modelo RoBERTa-base (MarIA) de forma eficiente.
