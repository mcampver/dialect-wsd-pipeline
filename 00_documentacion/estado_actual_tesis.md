# Resumen de Progreso: Detección Automática de Cubanismos

Este documento resume el progreso metodológico, experimental y de ingeniería realizado en el marco de la investigación para detectar y extraer cubanismos en textos de lenguaje natural.

---

## Fase 1: Digitalización y Estructuración del Diccionario Estandarizado
**Objetivo:** Transformar los documentos PDF del diccionario en una base de datos procesable para la investigación.
*   **Logros:**
    *   Se extrajo el texto de `CUBA1.pdf` y `CUBA2.pdf` mediante `PyMuPDF`.
    *   Se desarrolló un parser robusto con Expresiones Regulares para capturar: Lema, Categoría Gramatical, Definición y Ejemplos de equivalencia (ignorando cabeceras, paginación y saltos de línea irregulares).
    *   Se creó la base de datos `diccionario_cubanismos.db` (SQLite) importando con éxito **5,749 registros estructurados**.

## Fase 2: Construcción del Corpus de Evaluación (Ground Truth)
**Objetivo:** Obtener un *dataset* representativo de textos reales y generar un estándar de oro (Ground Truth) etiquetado.
*   **Logros:**
    *   Se recolectaron corpus lingüísticos (Cartas CORESPUC y archivo del diario Granma 2008-2019), totalizando **4,036 archivos**.
    *   Se desarrolló un script de **supervisión débil (Pre-etiquetado)** que buscó coincidencias crudas de los lemas del diccionario en el corpus, originando un CSV con **15,165 candidatos a cubanismo** en sus oraciones de contexto.
    *   **Anotación Conjunta Mixta:** A través de la revisión manual y la asistencia de una IA local (Ollama / Qwen), se consolidó un subconjunto etiquetado de **2,212 ejemplos** (`1` = cubanismo en contexto, `0` = falso positivo por polisemia).

## Fase 3: Evaluación del Modelo Base (Reglas y NLP con spaCy)
**Objetivo:** Establecer una línea base (*baseline*) que demuestre la insuficiencia del enfoque de solo diccionario y cuantifique el problema de la ambigüedad sintáctica.
*   **Logros:**
    *   Se integró `spaCy` (`es_core_news_md`) para el análisis morfosintáctico de los candidatos, extrayendo la etiqueta POS (Part-of-Speech) y la dependencia de las palabras en contexto.
    *   **Baseline 1 (Solo Diccionario):** Obtuvo una Precisión del **7%** y **2,042 falsos positivos**. (Demuestra que casi toda la búsqueda cruda es polisemia).
    *   **Baseline 2 (Diccionario + Regla Sintáctica de spaCy):** Filtró candidatos cuya categoría gramatical contextual no coincidía con la del diccionario. Mejoró la precisión sutilmente (**9%**).

## Fase 4 (En progreso): Experimentación con Machine Learning Predictivo
**Objetivo:** Entrenar verdaderos clasificadores que aprendan de la distribución semántica del texto usando características *TF-IDF* y *POS* para desambiguar.
*   **Logros Actuales:**
    *   Se implementó un modelo de **Machine Learning Tradicional (Random Forest)** que combina One-Hot Encoding de etiquetas spaCy con vectores TF-IDF de contexto.
    *   **Métricas Parciales:** La precisión subió radicalmente al **31%**, reduciendo dramáticamente los falsos positivos experimentados en la Fase 3, pero sufriendo en el *Recall* (14%) debido al fortísimo desbalance entre falsos candidatos y verdaderos cubanismos en el corpus de prueba.

---

## Fase 4: Experimento Zero-Shot (Ollama) - Hallazgo Crítico
*   **Modelo Evaluado:** `Qwen 3.5:2b` (Local).
*   **Metodología:** Inyección de definición real del Diccionario (RAG básico).
*   **Resultados:** **0% Recall**. El modelo no identificó ni uno solo de los 29 cubanismos del test set.
*   **Interpretación:** Este resultado es un **éxito metodológico**. Demuestra que los LLMs generalistas, a pesar de tener la definición lingüística a su disposición, sufren de un sesgo masivo hacia el español estándar y no poseen la capacidad de desambiguación semántica fina necesaria para el español de Cuba sin entrenamiento específico.

---

## 🚀 Fase 5: El Estado del Arte - Fine-Tuning de MarIA

Tras demostrar la insuficiencia de los modelos base y baselinas léxicas, el paso final es el **Fine-Tuning**.
1. **Modelo Seleccionado:** `MarIA` (`roberta-base-bne`), modelo entrenado por el BSC con la BNE, referente en español.
2. **Dataset:** 2,212 ejemplos validados con marcas de atención `[TGT]`.
3. **Objetivo:** Inyectar el conocimiento dialectal cubano en la arquitectura de un Transformer bidireccional para alcanzar el máximo rendimiento en desambiguación contextual.
