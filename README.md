# "Detección Automática de Cubanismos en Corpus Lingüísticos"

Este repositorio contiene el código fuente, los datos y los resultados del proyecto de tesis centrado en la detección de **cubanismos** (términos del español dialectal de Cuba) en textos de lenguaje natural.

El problema se aborda como una tarea de **Desambiguación de Sentido de Palabra (WSD - Word Sense Disambiguation)**. El sistema busca distinguir cuándo una palabra está siendo usada con su acepción dialectal cubana y cuándo con su acepción común en español estándar.

---

## 📋 Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas principales:

- `00_documentacion/`: Documentación del proyecto, resúmenes, estado actual e instrucciones para reproducir experimentos en Google Colab.
- `01_preprocesamiento/`: Scripts para la extracción de datos (pdfs, diccionarios), limpieza, pre-etiquetado, parsing con spaCy y división de los datos en conjuntos de entrenamiento (`train`) y evaluación (`test`).
- `02_experimentos/`: Código de todos los enfoques de modelación evaluados:
  - **Fase 3:** Baselines basados en diccionarios y reglas lingüísticas (spaCy).
  - **Fase 4:** Modelos de Machine Learning Clásico (Regresión Logística, Random Forest, SVM) usando TF-IDF.
  - **Fase 5:** Fine-tuning de Transformers (BETO, RoBERTa, DeBERTa, BERTIN, etc.).
  - **Fase 6 y 7:** Experimentos Zero-Shot con LLMs (Qwen) y QLoRA.
- `03_datos/`: Datasets utilizados en el proyecto (diccionarios estructurados, splits de train/test y muestras).
- `04_resultados/`: Archivos CSV iterativos con las métricas de evaluación obtenidas en cada fase y los scripts de auditoría y comparativa final.
- `06_aplicacion/`: Código para la inferencia y despliegue del clasificador final resultante de la investigación.

---

## 🚀 Metodología

La investigación sigue una progresión empírica para establecer qué familia de modelos resuelve mejor el problema de detección dialectal:

1. **Creación del Ground Truth**: A partir de diccionarios lexicográficos y corpus de texto cubanos (Cartas CORESPUC, Granma).
2. **Partición Canónica**: Un split estricto de 80/20 (`split_train.csv` y `split_test.csv`) para garantizar que la evaluación sea justa a través de todas las familias algorítmicas. El test set cuenta con 438 ejemplos (29 cubanismos / 409 no cubanismos).
3. **Baselines Lingüísticos**: Enfoques sin entrenamiento estadístico (búsquedas directas en diccionarios y filtros POS/DEP).
4. **Machine Learning Clásico**: Experimentación con bag-of-words (TF-IDF) y clasificadores tradicionales combinados con técnicas como SMOTE para el desbalance de clases.
5. **Transformers Encoder-Only**: Fine-tuning enfocado en *Sentence-Pair Classification* utilizando modelos del estado del arte preentrenados en español.
6. **LLMs (Zero-Shot y QLoRA)**: Exploración del uso de modelos generativos grandes para resolver la tarea WSD de forma directa y parametrizada eficiente.

---

## 🛠️ Requisitos e Instalación

Para ejecutar localmente el pipeline de datos o los experimentos clásicos:

```bash
# Clonar el repositorio
git clone https://github.com/mcampver/dialect-wsd-pipeline.git
cd dialect-wsd-pipeline

# Crear un entorno virtual (recomendado)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo de lenguaje de spaCy para español
python -m spacy download es_core_news_md
```

> **Nota:** Los experimentos de las Fases 5, 6 y 7 (Transformers y LLMs) están diseñados para ser ejecutados preferentemente en entornos con aceleración por GPU (p. ej. Google Colab / Kaggle). Revisa la carpeta `00_documentacion/` para guías detalladas.

---

## 📊 Evaluación y Resultados

La métrica principal de optimización es el **F1-Score para la clase minoritaria (Cubanismo)**, dado el alto desbalance de las clases (93% vs 6%). Se priorizó minimizar los *Falsos Positivos* propios de vocabularios polisémicos, maximizando al mismo tiempo el *Recall*. Todos los modelos fueron comparados exactamente sobre la misma partición del test original.

Revisa los reportes en `04_resultados/` y la tabla generada por los scripts visualizadores para el análisis a detalle de cada solución.

---

## 📝 Licencia

Este proyecto fue desarrollado bajo el marco de una tesis de grado. El código y los recursos generados se comparten con fines de investigación académica en Procensamiento de Lenguaje Natural (PLN). Consulte el archivo de Licencia para más detalles.
