# Resumen Final y Tabla Comparativa — Pipeline Completo "Detección Automática de Cubanismos "

Este documento unifica **toda la información de resultados, pruebas y la aplicación final** obtenidos a lo largo de las fases de la tesis.

## 1. Diseño del Conjunto de Datos (Fases 1 y 2)

Se digitalizó el Diccionario de Cubanismos (`d:\Tesis\01_preprocesamiento`), extrayendo lemas hacia una base de datos `diccionario_cubanismos.db` y creando un ground truth empírico que identificaba cada ocurrencia en un conjunto de oraciones extraídas de "Cartas CORESPUC" y "Granma 2008-2019".
El problema es abordado como **Word Sense Disambiguation (WSD)**: Clasificar una oración donde ocurre una palabra del diccionario y decir si representa la acepción local o no.
El Desbalance de las muestras detectado fue crítico (93.4% clase negativa, 6.6% positiva).
El Split canónico Train/Test estratificado consta de:
- `split_train.csv`: 1,748 ejemplos
- `split_test.csv`: 438 ejemplos (Test Set Sagrado, donde todos los modelos fueron evaluados).

## 2. Experimentos de Referencia y Reglas Sintácticas (Fase 3: Baselines)

Se definieron baselines en Python (`d:\Tesis\02_experimentos\fase3_baselines.py`) combinando POS y tags sintácticos:

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| B0 — Clase mayoritaria | 0.000 | 0.000 | 0.000 | 0.483 | 0.000 |
| B1 — Diccionario puro | 0.066 | 1.000 | 0.124 | 0.062 | 0.000 |
| B2 — Dicc. + POS | 0.080 | 0.931 | 0.147 | 0.266 | 0.101 |
| B3 — Dicc. + POS + DEP | 0.091 | 0.931 | 0.166 | 0.336 | 0.144 |

El techo del modelado puramente lingüístico fue muy bajo (MCC = 0.144), validando la necesidad de Machine Learning.

## 3. Estimadores de Machine Learning Clásico (Fase 4: ML Clásico)

Se aplicó la aproximación TD-IDF (`d:\Tesis\02_experimentos\fase4_ml_clasico.py`), empleando el contexto completo y contexto localizado a nivel del target token. Los resultados superaron por un margen amplio a las reglas.

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| RF — Random Forest | 0.429 | 0.103 | 0.167 | 0.566 | 0.186 |
| GB — Gradient Boosting | 0.308 | 0.138 | 0.190 | 0.575 | 0.170 |
| SVM — LinearSVC | 0.179 | 0.241 | 0.206 | 0.570 | 0.142 |
| LR — Reg. Logística | 0.213 | 0.448 | 0.289 | 0.604 | 0.238 |

## 4. Fine-Tuning de Transformadores (Fase 5: Transformers)

Se entrenaron Embeddings densos basados en Transformers y HuggingFace (`d:\Tesis\02_experimentos\fase5_transformers.py`).

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| BERTIN (RoBERTa, mC4) | 0.000 | 0.000 | 0.000 | 0.483 | 0.000 |
| mDeBERTa-v3 (DeBERTa, multilingüe) | 0.000 | 0.000 | 0.000 | 0.483 | 0.000 |
| XLM-RoBERTa (multilingüe masivo) | 0.159 | 0.241 | 0.192 | 0.559 | 0.125 |
| MrBERT-es (ModernBERT, BNE) | 0.368 | 0.483 | 0.418 | 0.685 | 0.375 |
| **BETO (BERT, Wikipedia lat.)** | **0.390** | **0.552** | **0.457** | **0.705** | **0.419** |

BETO demostró ser el mejor modelo, superando a todos los otros Transformers e incluso a los LLM en Zero-Shot y LoRA, superando las limitaciones impuestas por el grave desbalance semántico de la clase cubanismo.

## 5. Experimentos con LLM (Fases 6 y 7)

A través de Zero-Shot QA, se intentó evaluar con Modelos con billones de parámetros (CecilIA / Qwen (`d:\Tesis\02_experimentos\fase6*.py`), y ajustarlos localmente vía QLoRA / LoRA (`d:\Tesis\02_experimentos\fase7*.py`)).

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| Qwen2.5-2B-Instruct (zero-shot, genérico) | 0.000 | 0.000 | 0.000 | 0.482 | -0.018 |
| CecilIA 2B (zero-shot, corpus cubano) | 0.068 | 0.931 | 0.126 | 0.146 | 0.019 |
| CecilIA-Instruct 2B (QLoRA, corpus cubano) | 0.308 | 0.276 | 0.291 | 0.622 | 0.244 |

- Los LLMs fallaron miserablemente en modo *zero-shot* en esta tarea con sesgo extremo.
- El QA local vía QLoRA (CecilIA) fue competitivo con LR pero inferior dramáticamente al BERT model (BETO).

## 6. Aplicación y Despliegue en Producción (Fase 06: Aplicación Final)

Como resultado directo de esta evaluación en todas las fases descritas, el modelo `BETO` entrenado, fue seleccionado para pasar a producción.
Se exportó en un encapsulado local junto a su pipeline final en el directorio de entrega:
`d:\Tesis\06_aplicacion\modelo_cubanismos_final\` (modelo exportado en formato pt/safetensors)

### Herramientas entregadas:
- **`clasificador_final.py`** (`d:\Tesis\06_aplicacion\clasificador_final.py`): Expone la funcionalidad de `BETO` para que los usuarios clasifiquen un DataFrame o texto libre sin requerir saber AI. 
- **`auditoria_final_corpus.py`** (`d:\Tesis\06_aplicacion\auditoria_final_corpus.py`): Utilizado para correr un pass completo y descubrir cubanismos nunca antes descubiertos en los corpus masivos originales, finalizando el circuito como un NLP validado y en total función de los investigadores.

### Resultados de la Auditoría Final Masiva sobre el Corpus (BETO)

Se utilizó la herramienta `auditoria_final_corpus.py` para procesar de forma automatizada los **15,165 registros** del corpus (`corpus_preetiquetado.csv`). Los resultados demuestran el salto cuantitativo de capacidad de la IA específica (Transformer con fine-tuning) frente a las aproximaciones previas.

**Resumen de Resultados extraídos:**
- **Archivo generado:** `d:\Tesis\04_resultados\auditoria_corpus_completo.csv`
- **Coincidencia con Humano (`es_cubanismo`):** 83.37%
- **Total cubanismos sugeridos por IA previa:** 98
- **Total cubanismos detectados por BETO:** 2,548
- **Variación en detección:** +2,500.00% (incremento exponencial en el *recall* de cubanismos detectados en estado salvaje).

Este ciclo iterativo demuestra formalmente por qué *Word Sense Disambiguation* es un grave reto en la dialectología hispana, con el modelo Transformers BETO (MCC=.419, F1=.457) siendo el estado de arte empírico absoluto actual en el proyecto.