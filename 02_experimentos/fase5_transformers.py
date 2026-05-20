"""
fase5_transformers.py
=====================
Fine-tuning de BETO y MarIA para detección de cubanismos.
Diseñado para ejecutarse en Google Colab (GPU T4).

Tarea: Clasificación Binaria WSD (Word Sense Disambiguation)
  Input:  [CLS] oración completa [SEP] palabra_objetivo [SEP]
  Output: 0 (No cubanismo) | 1 (Cubanismo)

Estrategia de splits (sin data leakage):
  split_train.csv  (80% del total)
    ├── train_inner  (85%) → entrenamiento real
    └── val_inner    (15%) → eval_dataset del Trainer (selección de checkpoint)
  split_test.csv   (20% del total) → evaluación final, UNA SOLA VEZ, al terminar

  El test set sagrado NUNCA es visto por el Trainer durante el entrenamiento.
  Solo se toca al final con trainer.predict(), una vez elegido el mejor checkpoint.

Manejo del desbalance (93/7): weighted CrossEntropyLoss
  weight_clase1 = n_negativos / n_positivos del train_inner

Métricas reportadas: Precision, Recall, F1 (clase 1), F1-macro, MCC
"""

# ============================================================
# CELDA 1 — Instalación de dependencias (descomentar en Colab)
# ============================================================
# !pip install transformers datasets evaluate accelerate scikit-learn -q

import os, shutil
try:
    from google.colab import files
except ImportError:
    files = None
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ── Configuración ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MAX_LENGTH   = 128

# ── Modelos a comparar ─────────────────────────────────────────────────────
# Cada entrada: {"model_id": ..., "lr": ...}
# Se puede ajustar el LR por modelo según su comportamiento de convergencia.
MODELS = {
    # Encoders monolingües en español
    "BETO":        {"model_id": "dccuchile/bert-base-spanish-wwm-cased",          "lr": 2e-5},
    # "MrBERT-es":   {"model_id": "BSC-LT/MrBERT-es",                             "lr": 2e-5},
    # # LR reducido: BERTIN (mC4) converge más lento en dominios especializados
    # "BERTIN":      {"model_id": "bertin-project/bertin-roberta-base-spanish",    "lr": 1e-5},
    # # Encoders multilingües
    # # LR reducido: DeBERTa generalmente más estable con LR más conservador
    # "mDeBERTa-v3": {"model_id": "microsoft/mdeberta-v3-base",                   "lr": 1e-5},
    # "XLM-RoBERTa": {"model_id": "xlm-roberta-base",                             "lr": 2e-5},
}

# ── Reanudación parcial (para no repetir modelos ya completados) ──────────────
# Dejar vacío {} para entrenar todos los modelos desde cero.
# Rellenar con los alias de los modelos ya completados para saltarlos.
SKIP_COMPLETED = {}

# Modelos que requieren add_prefix_space=True en el tokenizador.
# Solo aplica a arquitecturas RoBERTa/BERTIN — NO a DeBERTa ni XLM-R.
ROBERTA_MODELS = {"bertin-project/bertin-roberta-base-spanish",
                  "BSC-TeMU/roberta-base-bne",
                  "BSC-LT/roberta-base-bne",
                  "BSC-LT/MrBERT-es"}

# Rutas flexibles (funciona en local y en Colab)
BASE_DIR = "/content" if os.path.exists("/content") else "."
TRAIN_CSV = os.path.join(BASE_DIR, "split_train.csv")
TEST_CSV  = os.path.join(BASE_DIR, "split_test.csv")
# ──────────────────────────────────────────────────────────────────────────────

# Semilla global para reproducibilidad
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CubanismosDataset(TorchDataset):
    """
    Tokeniza pares (oración, palabra_objetivo) para clasificación binaria.
    El tokenizador recibe:
      text      = oración completa  (contexto)
      text_pair = palabra objetivo   (qué desambiguar)
    Esto genera: [CLS] oración [SEP] palabra [SEP]
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            df['oracion'].tolist(),
            df['palabra'].tolist(),
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
        self.labels = df['y_true'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Trainer con pérdida ponderada ─────────────────────────────────────────────

class ImbalancedTrainer(Trainer):
    """
    Trainer con CrossEntropyLoss ponderada para compensar el desbalance 93/7.
    El peso de la clase minoritaria (cubanismo) = n_negativos / n_positivos.
    """

    def __init__(self, class_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Usar TANTO device COMO dtype del logit para evitar el error
        # "expected scalar type Half but found Float" en Colab con mixed precision (FP16)
        weight = self.class_weight.to(device=logits.device, dtype=logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    p, r, f1_cub, _ = precision_recall_fscore_support(
        labels, preds, pos_label=1, average='binary', zero_division=0
    )
    f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
    mcc    = matthews_corrcoef(labels, preds)

    return {
        "precision_cub": round(float(p), 4),
        "recall_cub":    round(float(r), 4),
        "f1_cubanismo":  round(float(f1_cub), 4),  # ← checkpoint metric
        "f1_macro":      round(float(f1_mac), 4),
        "mcc":           round(float(mcc), 4),
    }


# ── Entrenamiento y evaluación de un modelo ───────────────────────────────────

def entrenar_y_evaluar(alias: str, model_id: str, lr: float,
                       df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"Procesando: {alias}  ({model_id})")
    print(f"{'='*60}")

    # ── Split interno: train_inner (85%) + val_inner (15%) ────────────────────
    # El val_inner se usa SOLO para seleccionar el mejor checkpoint.
    # El df_test (test sagrado) NUNCA entra al Trainer.
    df_train_inner, df_val_inner = train_test_split(
        df_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df_train['y_true'],
    )
    print(f"  train_inner : {len(df_train_inner)} ej. | "
          f"{df_train_inner['y_true'].sum()} cubanismos")
    print(f"  val_inner   : {len(df_val_inner)} ej. | "
          f"{df_val_inner['y_true'].sum()} cubanismos  (solo para checkpoint)")
    print(f"  test sagrado: {len(df_test)} ej. | "
          f"{df_test['y_true'].sum()} cubanismos  (evaluacion final, 1 sola vez)")

    # Tokenizador: add_prefix_space solo para arquitecturas RoBERTa/BERTIN
    # DeBERTa y XLM-RoBERTa usan SentencePiece y no aceptan ese parámetro
    needs_prefix = model_id in ROBERTA_MODELS
    tokenizer_kwargs = {"add_prefix_space": True} if needs_prefix else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

    # Datasets — el test set se tokeniza pero NO entra al Trainer
    ds_train_inner = CubanismosDataset(df_train_inner, tokenizer, MAX_LENGTH)
    ds_val_inner   = CubanismosDataset(df_val_inner,   tokenizer, MAX_LENGTH)
    ds_test        = CubanismosDataset(df_test,        tokenizer, MAX_LENGTH)

    # Peso de clase minoritaria calculado sobre train_inner únicamente
    n_neg = (df_train_inner['y_true'] == 0).sum()
    n_pos = (df_train_inner['y_true'] == 1).sum()
    weight_pos = n_neg / n_pos
    print(f"  Peso clase 1 (cubanismo): {weight_pos:.2f}")
    class_weight = torch.tensor([1.0, weight_pos], dtype=torch.float)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "O", 1: "CUB"},
        label2id={"O": 0, "CUB": 1},
        ignore_mismatched_sizes=True,
    )

    output_dir = f"./modelo_{alias.lower()}"

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_cubanismo",  # F1 de clase cubanismo, no accuracy
        greater_is_better=True,
        logging_steps=20,
        seed=RANDOM_STATE,
        report_to="none",
    )

    # El Trainer ve SOLO ds_train_inner y ds_val_inner — nunca el test set
    trainer = ImbalancedTrainer(
        class_weight=class_weight,
        model=model,
        args=args,
        train_dataset=ds_train_inner,   # entrenamiento real
        eval_dataset=ds_val_inner,      # seleccion de checkpoint (val interno)
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Mejor checkpoint ya cargado (load_best_model_at_end=True)

    # GUARDAR MODELO SI ES BETO
    if alias == "BETO":
        print(f"\n[EXPORT] Guardando modelo final de {alias}...")
        trainer.save_model("./modelo_cubanismos_final")
        tokenizer.save_pretrained("./modelo_cubanismos_final")

    # ── Evaluacion final sobre el test sagrado (UNA SOLA VEZ) ─────────────────
    preds_out = trainer.predict(ds_test)
    y_pred    = np.argmax(preds_out.predictions, axis=1)
    y_true    = df_test['y_true'].astype(int).tolist()

    # Calcular métricas finales directamente (no desde el Trainer — ese usa val_inner)
    p, r, f1_cub, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)

    # Limpieza de GPU
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  [RESULTADO FINAL — TEST SET SAGRADO]")
    print(f"  Precision : {p:.4f} | Recall: {r:.4f}")
    print(f"  F1 (CUB)  : {f1_cub:.4f} | F1-macro: {f1_mac:.4f} | MCC: {mcc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        'Modelo':        alias,
        'Precision_CUB': round(float(p), 4),
        'Recall_CUB':    round(float(r), 4),
        'F1_CUB':        round(float(f1_cub), 4),
        'F1_macro':      round(float(f1_mac), 4),
        'MCC':           round(float(mcc), 4),
        'TP': int(cm[1,1]), 'FP': int(cm[0,1]),
        'TN': int(cm[0,0]), 'FN': int(cm[1,0]),
        'y_pred':        y_pred.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤗 FASE 5: FINE-TUNING DE TRANSFORMERS (WSD Binaria)")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, sep=';', encoding='utf-8-sig')
    df_test  = pd.read_csv(TEST_CSV,  sep=';', encoding='utf-8-sig')

    df_train['oracion'] = df_train['oracion'].fillna('').astype(str)
    df_train['palabra'] = df_train['palabra'].fillna('').astype(str)
    df_test['oracion']  = df_test['oracion'].fillna('').astype(str)
    df_test['palabra']  = df_test['palabra'].fillna('').astype(str)

    print(f"Train: {len(df_train)} | Cubanismos: {df_train['y_true'].sum()}")
    print(f"Test : {len(df_test)} | Cubanismos: {df_test['y_true'].sum()}")

    resultados = []

    for alias, cfg in MODELS.items():
        if alias in SKIP_COMPLETED:
            print(f"[SKIP] {alias} — ya completado en ejecucion anterior")
            continue
        model_id = cfg["model_id"]
        lr       = cfg["lr"]
        res = entrenar_y_evaluar(alias, model_id, lr, df_train, df_test)
        resultados.append(res)

        df_out = df_test.copy()
        df_out[f'pred_{alias}'] = res['y_pred']
        df_out.to_csv(f"resultados_fase5_{alias}.csv", sep=';',
                      index=False, encoding='utf-8-sig')
        # === NUEVO CÓDIGO PARA EXTRAER ERRORES Y ACIERTOS (PARA EL CAPÍTULO 7) ===
        if alias == "BETO":
            # 1. Falsos Positivos: El modelo dijo 1 (Cubanismo), pero era 0 (Estándar)
            fp_df = df_out[(df_out['y_true'] == 0) & (df_out[f'pred_{alias}'] == 1)]
            fp_df.to_csv("BETO_Falsos_Positivos.csv", sep=';', index=False, encoding='utf-8-sig')

            # 2. Falsos Negativos: El modelo dijo 0 (Estándar), pero era 1 (Cubanismo)
            fn_df = df_out[(df_out['y_true'] == 1) & (df_out[f'pred_{alias}'] == 0)]
            fn_df.to_csv("BETO_Falsos_Negativos.csv", sep=';', index=False, encoding='utf-8-sig')

            # 3. Verdaderos Positivos: El modelo dijo 1 y era 1 (Para mostrar ejemplos buenos)
            tp_df = df_out[(df_out['y_true'] == 1) & (df_out[f'pred_{alias}'] == 1)]
            tp_df.to_csv("BETO_Verdaderos_Positivos.csv", sep=';', index=False, encoding='utf-8-sig')

            print(f"\n[EXPORT] Archivos de análisis (FP, FN, TP) guardados para la tesis.")
        

    # Tabla comparativa final
    df_comp = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred'}
                            for r in resultados])
    print(f"\n{'='*60}")
    print("🏆 TABLA COMPARATIVA FINAL — TRANSFORMERS")
    print(df_comp[['Modelo','Precision_CUB','Recall_CUB',
                   'F1_CUB','F1_macro','MCC']].to_string(index=False))

    df_comp.to_csv("comparativa_fase5.csv", sep=';', index=False)
    print("\n💾 Tabla guardada en: comparativa_fase5.csv")

    # --- DESCARGAR MODELO (COLAB) ---
    if os.path.exists("./modelo_cubanismos_final"):
        print("\n📦 Comprimiendo modelo para descarga...")
        shutil.make_archive("modelo_cubanismos", 'zip', "./modelo_cubanismos_final")
        if files:
            print("🚀 Intentando iniciar descarga automática de modelo_cubanismos.zip...")
            try:
                files.download('modelo_cubanismos.zip')
            except AttributeError:
                print("⚠️ La descarga automática falló (normal al ejecutar con !python).")
                print("👉 Por favor, descarga 'modelo_cubanismos.zip' manualmente desde el panel de Archivos (icono de carpeta a la izquierda).")
        else:
            print("💾 El modelo está listo en: modelo_cubanismos.zip")

if __name__ == "__main__":
    main()
