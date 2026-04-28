# Instrucciones Exactas — Fase 5 en Google Colab
## Fine-Tuning BETO + BSC-RoBERTa-BNE para Detección de Cubanismos

---

## Archivos incluidos en este ZIP

| Archivo | Descripción |
|---|---|
| `split_train.csv` | 1,748 ejemplos de entrenamiento (80% del ground truth) |
| `split_test.csv` | 438 ejemplos de prueba sagrados (20% — solo evaluación final) |
| `fase5_transformers.py` | Script de fine-tuning completo |

---

## PASO 1 — Abrir Google Colab y configurar GPU

1. Ve a [https://colab.research.google.com](https://colab.research.google.com)
2. Crea un nuevo notebook: **Archivo → Nuevo cuaderno**
3. Cambia el runtime a GPU:
   - **Entorno de ejecución → Cambiar tipo de entorno de ejecución**
   - Acelerador de hardware: **GPU**
   - Tipo de GPU: **T4** (la gratuita es suficiente)
   - Clic en **Guardar**
4. Verifica que la GPU esté activa ejecutando esta celda:

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Salida esperada:**
```
GPU disponible: True
GPU: Tesla T4
```

> ⚠️ Si dice `False`, el runtime no tiene GPU. Repite el paso 3.

---

## PASO 2 — Subir el ZIP a Colab

**Opción A (recomendada): Subida directa**

Ejecuta esta celda para subir el archivo:

```python
from google.colab import files
files.upload()
# Se abrirá un diálogo de selección de archivos
# Sube: tesis_fase5_colab.zip
```

**Opción B: Desde Google Drive**

Si ya tienes el ZIP en Drive, monta Drive y copia el archivo:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/drive/MyDrive/tesis_fase5_colab.zip', '/content/')
```

---

## PASO 3 — Descomprimir el ZIP

```python
import zipfile, os

with zipfile.ZipFile('/content/tesis_fase5_colab.zip', 'r') as z:
    z.extractall('/content/')

# Verificar que los archivos están presentes
archivos = os.listdir('/content/')
print("Archivos en /content/:")
for f in sorted(archivos):
    if f.endswith(('.csv', '.py')):
        print(f"  ✓ {f}")
```

**Salida esperada:**
```
Archivos en /content/:
  ✓ fase5_transformers.py
  ✓ split_test.csv
  ✓ split_train.csv
```

---

## PASO 4 — Instalar dependencias

```python
!pip install transformers>=4.41.0 accelerate scikit-learn sentencepiece -q
```

> ⏱️ Tarda ~2 minutos la primera vez.
> 
> `sentencepiece` es requerido por el tokenizador de `mDeBERTa-v3` (arquitectura DeBERTa de Microsoft).
> 
> ⚠️ Si aparece el mensaje `WARNING: Running pip as the 'root' user...` es normal, no es un error.

---

## PASO 5 — Ejecutar el entrenamiento

```python
!python /content/fase5_transformers.py
```

### ¿Qué verás durante la ejecución?

El script entrena **dos modelos secuencialmente** (BETO y luego BSC-RoBERTa-BNE):

```
============================================================
FASE 5: FINE-TUNING DE TRANSFORMERS (WSD Binaria)
============================================================
Train: 1748 | Cubanismos: 115
Test : 438  | Cubanismos: 29

============================================================
Procesando: BETO  (dccuchile/bert-base-spanish-wwm-cased)
============================================================
  train_inner : 1485 ej. | 97 cubanismos
  val_inner   : 263 ej.  | 18 cubanismos  (solo para checkpoint)
  test sagrado: 438 ej.  | 29 cubanismos  (evaluacion final, 1 sola vez)
  Peso clase 1 (cubanismo): 15.31

[Epoch 1/5] loss: 0.XXXX | val f1_cubanismo: 0.XXXX
[Epoch 2/5] ...
...

  [RESULTADO FINAL — TEST SET SAGRADO]
  Precision : X.XXXX | Recall: X.XXXX
  F1 (CUB)  : X.XXXX | F1-macro: X.XXXX | MCC: X.XXXX
  TN=XXX  FP=XX  FN=XX  TP=XX

============================================================
Procesando: BSC-RoBERTa-BNE  (BSC-LT/roberta-base-bne)
============================================================
  [... mismo proceso ...]
```

### Tiempos estimados en GPU T4

| Modelo | Arquitectura | Tiempo estimado |
|--------|-------------|-----------------|
| BETO | BERT | ~20–30 min |
| MrBERT-es | ModernBERT | ~20–30 min |
| BERTIN | RoBERTa | ~20–30 min |
| mDeBERTa-v3 | DeBERTa | ~30–40 min |
| XLM-RoBERTa | RoBERTa multilingüe | ~20–30 min |
| **Total** | | **~110–160 minutos** |

> ⚠️ Con la sesión gratuita de Colab (~12h de GPU disponibles) tienes tiempo de sobra para los 5 modelos en una misma sesión.

> ⚠️ **Importante:** No cierres la pestaña de Colab durante el entrenamiento. Si el runtime se desconecta, tendrás que volver a ejecutar desde el PASO 5. El modelo no se guarda automáticamente fuera del runtime.

---

## PASO 6 — Verificar los archivos de resultado

```python
import os, pandas as pd

# Archivos de resultados generados
resultados = [f for f in os.listdir('/content/') if f.startswith('comparativa') or f.startswith('resultados_fase5')]
print("Archivos de resultados:")
for f in sorted(resultados):
    print(f"  ✓ {f}")

# Mostrar la tabla comparativa final
df = pd.read_csv('/content/comparativa_fase5.csv', sep=';')
print("\n=== TABLA COMPARATIVA FINAL ===")
print(df[['Modelo','Precision_CUB','Recall_CUB','F1_CUB','F1_macro','MCC']].to_string(index=False))
```

---

## PASO 7 — Descargar los resultados

```python
from google.colab import files

# Descargar todos los archivos de resultados
archivos_a_descargar = [
    '/content/comparativa_fase5.csv',
    '/content/resultados_fase5_BETO.csv',
    '/content/resultados_fase5_BSC-RoBERTa-BNE.csv',
]

for ruta in archivos_a_descargar:
    if os.path.exists(ruta):
        files.download(ruta)
        print(f"Descargado: {ruta}")
    else:
        print(f"NO encontrado: {ruta}")
```

---

## Solución de problemas frecuentes

### Error: `CUDA out of memory`
Reduce el batch size editando el script antes de ejecutarlo:
```python
# En fase5_transformers.py, busca y cambia:
per_device_train_batch_size=16  →  per_device_train_batch_size=8
per_device_eval_batch_size=32   →  per_device_eval_batch_size=16
```

### Error: `eval_strategy is not supported` 
La versión de transformers es antigua. Ejecuta primero:
```python
!pip install transformers==4.45.0 -q
```
Luego reinicia el runtime: **Entorno de ejecución → Reiniciar sesión** y repite desde el PASO 4.

### El runtime se desconectó durante el entrenamiento
Vuelve a ejecutar desde el PASO 5. Los modelos parciales en `./modelo_beto/` y `./modelo_bsc-roberta-bne/` se pierden al reiniciar el runtime — no hay forma de reanudar.

---

## Qué hacer con los resultados

Una vez que tengas `comparativa_fase5.csv` descargado, los resultados se integran directamente en la **Tabla Comparativa Global** de la tesis:

| Modelo | F1 (CUB) | F1-macro | MCC |
|---|---|---|---|
| B3 — Dicc. + POS + DEP | 0.166 | 0.336 | 0.144 |
| LR — Regresión Logística | 0.289 | 0.604 | 0.238 |
| BETO fine-tuned | *tu resultado* | *tu resultado* | *tu resultado* |
| BSC-RoBERTa-BNE fine-tuned | *tu resultado* | *tu resultado* | *tu resultado* |
