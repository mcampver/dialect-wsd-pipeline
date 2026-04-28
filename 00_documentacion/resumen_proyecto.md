# Resumen Técnico-Científico del Proyecto

## "Detección Automática de Cubanismos en Corpus Lingüísticos"

### Sistema de PLN Completo — `d:\Tesis`

---

## 1. Objetivo General

Desarrollar un sistema capaz de detectar **cubanismos** (términos del español dialectal de Cuba) en textos de lenguaje natural, distinguiéndolos de sus acepciones comunes en español estándar — el clásico problema de **desambiguación de sentido de palabra (WSD)**.

El enfoque es progresivo: se parte de un diccionario léxico, se construye un _ground truth_ empírico, y se evalúan múltiples aproximaciones computacionales de menor a mayor sofisticación, todas comparadas sobre el mismo test set.

---

## 2. Framing Correcto del Problema

> **La tarea es WSD (Word Sense Disambiguation), NO NER.**

El ground truth contiene pares `(oración, palabra_objetivo)` con etiqueta binaria: ¿está esta palabra siendo usada como cubanismo en este contexto? El modelo necesita saber **cuál** palabra desambiguar. El enfoque de fine-tuning correcto es **clasificación binaria sentence-pair**, no etiquetado de secuencias BIO.

---

## 3. Resumen por Fases

---

### FASE 1 — Digitalización del Diccionario ✅

**Resultado:** `diccionario_cubanismos.db` — base de datos SQLite con **5,749 entradas** estructuradas.

```bash
python extract_to_db.py
```

---

### FASE 2 — Construcción del Ground Truth ✅

**Corpus:** 4,036 archivos de texto (Cartas CORESPUC + Granma 2008–2019).

**Proceso:**

1. Búsqueda de lemas del diccionario en el corpus → ~15,000 candidatos (`corpus_preetiquetado.csv`)
2. Anotación manual: ~300 ejemplos por el investigador
3. Anotación asistida por IA: ~1,900 ejemplos con `anotar_ia.py` (Qwen vía Ollama)
4. Corrección manual de errores de la IA

**Ground Truth Final:** `resultados_baseline.csv`

| Clase                | Ejemplos  | Proporción |
| -------------------- | --------- | ---------- |
| `0` (No cubanismo)   | 2,042     | 93.4%      |
| `1` (Cubanismo real) | 144       | 6.6%       |
| **Total**            | **2,186** |            |

> **Hallazgo clave:** El desbalance extremo (93.4/6.6) es un resultado académico: demuestra empíricamente la alta tasa de polisemia de los cubanismos en textos reales.

```bash
python pre_etiquetar.py   # genera corpus_preetiquetado.csv
# Luego anotar manualmente la columna y_true
```

---

### Split Canónico Train/Test (ejecutar antes de Fase 3)

```bash
python crear_split_canonico.py
```

**Resultado (estratificado 80/20, random_state=42):**

| Set               | Ejemplos | Cubanismos |
| ----------------- | -------- | ---------- |
| `split_train.csv` | 1,748    | 115 (6.6%) |
| `split_test.csv`  | 438      | 29 (6.6%)  |

> **Regla de oro:** `split_test.csv` es el **test set sagrado**. NINGÚN modelo lo usa durante el entrenamiento. Todos los modelos (Fases 3, 4 y 5) se evalúan sobre los mismos 438 ejemplos para garantizar comparación científicamente válida.

---

### FASE 3 — Baselines Lingüísticos ✅

**Problema:** ¿Cuánto logran aproximaciones puramente basadas en reglas?

**Script:** `fase3_baselines.py`

**Baselines implementados:**

- **B0:** Clase mayoritaria (predice siempre 0) — suelo absoluto
- **B1:** Diccionario puro (predice siempre 1) — upper bound de Recall
- **B2:** Diccionario + filtro POS (categoría gramatical contextual vs. diccionario)
- **B3:** Diccionario + POS + filtro DEP (función sintáctica esperada)

**Resultados en TEST SET (438 ejemplos, 29 cubanismos):**

| Modelo                 | Precision | Recall | F1 (CUB)  | F1-macro | MCC   |
| ---------------------- | --------- | ------ | --------- | -------- | ----- |
| B0 — Clase mayoritaria | 0.000     | 0.000  | 0.000     | 0.483    | 0.000 |
| B1 — Diccionario puro  | 0.066     | 1.000  | 0.124     | 0.062    | 0.000 |
| B2 — Dicc. + POS       | 0.080     | 0.931  | 0.147     | 0.266    | 0.101 |
| B3 — Dicc. + POS + DEP | 0.091     | 0.931  | **0.166** | 0.336    | 0.144 |

> **Hallazgos clave:**
>
> - La Precision del 6.6% de B1 confirma que solo 1 de cada 15 coincidencias con el diccionario es un cubanismo real en contexto.
> - El filtro POS mejora la Precision del 6.6% al 8%, pero no resuelve la ambigüedad semántica.
> - Añadir la dependencia sintáctica (B3) da la mejor combinación de reglas: F1=0.166, MCC=0.144.
> - El techo de las reglas lingüísticas es MCC≈0.14 — insuficiente para una aplicación real.

```bash
python fase3_baselines.py
```

---

### FASE 4 — Machine Learning Clásico ✅

**Problema:** ¿Puede un clasificador aprender patrones contextuales suficientes para superar las reglas?

**Script:** `fase4_ml_clasico.py`

**Arquitectura de features:**

- TF-IDF de la oración completa (max_features=3000, ngram_range=(1,2)) — contexto global
- TF-IDF de la ventana local ±5 tokens alrededor de la palabra objetivo — contexto local
- OneHot de: `spacy_pos`, `spacy_dep`, `cat_diccionario`

**Protocolo:** 5-Fold Stratified CV sobre `split_train.csv` para validar. Evaluación final en `split_test.csv`.

**Resultados en TEST SET:**

| Modelo                   | F1 (CUB)  | F1-macro  | MCC       | F1 CV (mean±std) |
| ------------------------ | --------- | --------- | --------- | ---------------- |
| LR — Regresión Logística | **0.289** | **0.604** | **0.238** | 0.217 ± 0.054    |
| SVM — LinearSVC          | 0.206     | 0.570     | 0.142     | 0.170 ± 0.047    |
| RF — Random Forest       | 0.167     | 0.566     | 0.186     | 0.000 ± 0.000    |
| GB — Gradient Boosting   | 0.190     | 0.575     | 0.170     | 0.075 ± 0.032    |

> **Hallazgos clave:**
>
> - La Regresión Logística supera a RF y GB: F1=0.289, MCC=0.238 (vs. MCC≈0.14 de los baselines de reglas).
> - El RF muestra F1_CV=0.000 en cross-validation pero F1=0.167 en test → indicativo de sobreajuste al threshold por defecto con datos desbalanceados.
> - El mejor modelo ML (LR) mejora el mejor baseline de reglas (B3) en un **+74% en F1** y **+65% en MCC**.
> - Ningún modelo ML supera el umbral de F1=0.30 — la ambigüedad semántica profunda requiere comprensión contextual bidireccional.

```bash
python fase4_ml_clasico.py
```

---

### FASE 5 — Fine-Tuning de Transformers ✅

**Problema:** ¿Puede un modelo preentrenado en español, ajustado sobre el corpus dialectal, aprender la desambiguación semántica?

**Script:** `fase5_transformers.py`

**Arquitectura:** Clasificación binaria sentence-pair.

**Resultados Destacados en TEST SET:**

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| BERTIN | 0.000 | 0.000 | 0.000 | 0.483 | 0.000 |
| XLM-RoBERTa | 0.159 | 0.241 | 0.192 | 0.559 | 0.125 |
| MrBERT-es | 0.368 | 0.483 | 0.418 | 0.685 | 0.375 |
| **BETO** | **0.390** | **0.552** | **0.457** | **0.705** | **0.419** |

> **Hallazgos clave:** BETO superó a todos los demás Transformers demostrando que la representación profunda mitigó severamente el desbalance semántico de la clase minoritaria.

---

### FASE 6 y 7 — Experimentos con LLM (Zero-Shot y QLoRA) ✅

**Problema:** Evaluación del conocimiento pre-entrenado y ajuste de modelos generativos de parámetros billonarios en tareas WSD complejas de dialectos (Qwen y CecilIA).

**Scripts:** `fase6_*.py` y `fase7_*.py`

| Modelo | Precision | Recall | F1 (CUB) | F1-macro | MCC |
|--------|-----------|--------|-----------|----------|-----|
| Qwen2.5-2B-Instruct (zero-shot) | 0.000 | 0.000 | 0.000 | 0.482 | -0.018 |
| CecilIA 2B (zero-shot) | 0.068 | 0.931 | 0.126 | 0.146 | 0.019 |
| CecilIA-Instruct 2B (QLoRA) | 0.308 | 0.276 | 0.291 | 0.622 | 0.244 |

> **Hallazgos clave:** Los LLMs, aún adaptados a instrucciones en español cubano, fallaron en *zero-shot* debido al profundo sesgo distribucional de cubanismos. Aunque se mejoró empleando QLoRA (al nivel de ML Clásico), quedaron muy por debajo de BETO.

---

### FASE 06 — Aplicación y Despliegue (Auditoría Final) ✅

**Directorio:** `06_aplicacion/`
Como corolario se seleccionó a `BETO` para empaquetarse en un sistema desplegable.

**Ejecución de auditoría masiva sobre todo el corpus (~15k de ejemplos):**
- Coincidencia con humano: 83.37%
- Cubanismos identificados por BETO en ambiente "salvaje": 2,548 frente a los escasos 98 iniciales, marcando una variación de detección del +2,500.00%.

## 4. Tabla Comparativa Global

| Modelo                   | Approach          | F1 (CUB)    | F1-macro    | MCC         |
| ------------------------ | ----------------- | ----------- | ----------- | ----------- |
| B0 — Clase mayoritaria   | Regla             | 0.000       | 0.483       | 0.000       |
| B3 — Dicc. + POS + DEP   | Regla sintáctica+ | 0.166       | 0.336       | 0.144       |
| LR — Regresión Logística | ML Clásico        | 0.289       | 0.604       | 0.238       |
| BETO fine-tuned          | Transformer       | **0.457**   | **0.705**   | **0.419**   |
| CecilIA-Instruct (QLoRA) | LLM QLoRA         | 0.291       | 0.622       | 0.244       |

> **Nota:** MCC (Matthews Correlation Coefficient) es la métrica más robusta. F1 (CUB) es la principal de la tarea. La supremacía de BETO indica que los Modelos del Lenguaje Preentrenados y finetuneados específicamente para Encoders, actúan como las mejores herramientas de WSD.

---

## 5. Inventario de Scripts (Versión Actual)

| Archivo                     | Rol                              | Estado |
| --------------------------- | -------------------------------- | ------ |
| `fase1` y `fase2`           | Digitalización y Etiquetado IA   | ✅     |
| `crear_split_canonico.py`   | Split train/test                 | ✅     |
| `fase3_baselines.py`        | Baselines de Reglas              | ✅     |
| `fase4_ml_clasico.py`       | Scikit-learn LR, SVG, RF, GB     | ✅     |
| `fase5_transformers.py`     | Fine-Tuning Transformers         | ✅     |
| `fase6_*` y `fase7_*`       | Pruebas con LLMs gen. y QLoRA    | ✅     |
| `06_aplicacion/`            | Export final (BETO) y scripts    | ✅     |

---

## 6. Hallazgos Académicos

1. **Polisemia Extrema (93.4%):** Solo 6.6% de coincidencias de diccionario son cubanismos reales.
2. **Techo Lingüístico (MCC=0.144):** Las reglas puras son ineficaces para WSD.
3. **ML Clásico (+65% MCC):** Extrae un patrón estadístico, con límite estructural.
4. **Resistencia de LLMs al WSD sin Tuning Fino Intensivo:** El Zero-Shot fracasó debido a un sesgo estricto hacia el español estándar, e inclusive con LoRA, se encarece demasiado con resultados mediocres por debajo del 30% en F1.
5. **Victoria de los Modelos Encoder (BETO):** Con un F1=0.457 y MCC=0.419 en el test set sagrado, un modelo *bert-based* latinoamericano es el único con capacidad representacional real de clasificar las acepciones con éxito moderado. Además, el alcance se amplificó empíricamente con un Pass del Corpus Completo incrementando +2500% el Recall automático sobre un pipeline manual.

---

## 7. Próximos Pasos

- [X] Compilar y actualizar toda la experimentación en documentación MD definitiva.
- [ ] Incorporar el análisis cuantitativo y los descubrimientos a la redacción oficial en látex/docx de la Tesis.
- [ ] Explorar vías de ampliación e integración del API de `clasificador_final.py` en otros proyectos humanísticos del equipo colegiado.
