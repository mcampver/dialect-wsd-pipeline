"""
fase6_cecilia_zeroshot.py
=========================
Experimento Zero-Shot con CecilIA 2B — LLM preentrenado en corpus cubano.

Hipótesis experimental:
  Un modelo con 1,000M tokens exclusivamente cubanos puede desambiguar
  cubanismos sin fine-tuning, superando a Qwen (LLM genérico, zero-shot).

Comparativa de referencia (mismo test set, 438 ejemplos):
  - Qwen 3.5:2b  (genérico, zero-shot)         → F1=0.000  MCC=0.000
  - BETO         (Wikipedia lat., fine-tuned)   → F1=0.457  MCC=0.419  [BETO]
  - CecilIA 2B   (corpus cubano, zero-shot)     → ?

Diferencia arquitectónica clave vs Fase 5:
  Fase 5: Encoder bidireccional  → AutoModelForSequenceClassification
  Fase 6: Decoder causal         → AutoModelForCausalLM + prompt

El modelo NUNCA fue fine-tuned para esta tarea.
El resultado mide cuánto aporta el preentrenamiento dialectal por sí solo.

⚠️  VERIFICAR EL MODEL_ID antes de ejecutar:
    Busca "CecilIA" en https://huggingface.co/models?language=es
    y actualiza MODEL_ID con el identificador oficial.
"""

# ==============================================================
# CELDA 1 — Instalación (descomentar en Colab)
# ==============================================================
# !pip install transformers accelerate bitsandbytes scikit-learn -q

import os
import re
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    f1_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix,
)

# ── Configuración ──────────────────────────────────────────────────────────────
# ⚠️  Verificar en HuggingFace: https://huggingface.co/models?search=cecilia+cuba
MODEL_ID = "CecilIA-LM/cecilia-2b-v0.1"   # ACTUALIZAR SI ES NECESARIO

RANDOM_STATE   = 42
MAX_NEW_TOKENS = 8      # Solo necesitamos "SÍ" o "NO"
BATCH_SIZE     = 1      # Inferencia una oración a la vez (más estable con LLMs)
CHECKPOINT_N   = 50     # Guardar resultados parciales cada N ejemplos

BASE_DIR  = "/content" if os.path.exists("/content") else "."
TEST_CSV  = os.path.join(BASE_DIR, "split_test.csv")
OUT_CSV   = os.path.join(BASE_DIR, "resultados_fase6_cecilia.csv")
CKPT_JSON = os.path.join(BASE_DIR, "cecilia_checkpoint.json")
# ──────────────────────────────────────────────────────────────────────────────

# Resultados previos de referencia (hardcoded para la tabla comparativa final)
RESULTADOS_REFERENCIA = [
    {"Modelo": "BETO (fine-tuned, Wikipedia lat.)",
     "F1_CUB": 0.457, "F1_macro": 0.705, "MCC": 0.419,
     "Precision_CUB": 0.390, "Recall_CUB": 0.552},
]


# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Eres un experto en lingüística del español de Cuba. \
Tu tarea es determinar si una palabra está siendo usada como cubanismo \
(expresión dialectal de Cuba) en una oración.

Oración: {oracion}
Palabra objetivo: {palabra}

Pregunta: ¿La palabra "{palabra}" está siendo usada como cubanismo en la oración anterior?
Responde únicamente con una sola palabra: SÍ o NO.
Respuesta:"""


def construir_prompt(oracion: str, palabra: str) -> str:
    return PROMPT_TEMPLATE.format(oracion=oracion.strip(), palabra=palabra.strip())


# ── Parseo de la respuesta generada ───────────────────────────────────────────

def parsear_respuesta(texto_generado: str) -> int:
    """
    Extrae 0 o 1 del texto generado por el modelo.
    Busca 'SI'/'SÍ' (→ 1) o 'NO' (→ 0) en los primeros caracteres generados.
    Si no puede parsear, devuelve 0 (conservador) y registra el caso.
    """
    # Limpiar y tomar solo los primeros 30 caracteres de la respuesta
    texto = texto_generado.strip()[:30].upper()

    # Eliminar puntuación inicial
    texto = re.sub(r'^[^A-ZÁÉÍÓÚÑ]+', '', texto)

    if texto.startswith('S'):   # SÍ, SI, Sí, Si
        return 1
    if texto.startswith('N'):   # NO, No, no
        return 0

    # Fallback: buscar en cualquier posición
    if re.search(r'\bS[IÍ]\b', texto):
        return 1
    if re.search(r'\bNO\b', texto):
        return 0

    return 0   # Default conservador


# ── Carga del modelo con cuantización 4-bit ───────────────────────────────────

def cargar_modelo():
    """Carga CecilIA 2B con cuantización NF4 para caber en T4 (16 GB VRAM)."""
    print(f"Cargando modelo: {MODEL_ID}")
    print("Usando cuantización 4-bit (NF4) para GPU T4...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Modelo cargado. Parametros: "
          f"~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return tokenizer, model


# ── Inferencia para un ejemplo ────────────────────────────────────────────────

def inferir(oracion: str, palabra: str, tokenizer, model) -> tuple[int, str]:
    """
    Devuelve (prediccion, texto_generado_raw).
    prediccion: 0 o 1
    texto_generado_raw: para diagnóstico y análisis de errores
    """
    prompt = construir_prompt(oracion, palabra)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # Greedy — reproducible
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extraer solo los tokens NUEVOS (no repetir el prompt)
    n_input_tokens = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, n_input_tokens:]
    texto_generado = tokenizer.decode(new_tokens, skip_special_tokens=True)

    prediccion = parsear_respuesta(texto_generado)
    return prediccion, texto_generado


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FASE 6: CECILIA ZERO-SHOT (Corpus Cubano)")
    print("=" * 60)

    df_test = pd.read_csv(TEST_CSV, sep=';', encoding='utf-8-sig')
    df_test['oracion'] = df_test['oracion'].fillna('').astype(str)
    df_test['palabra'] = df_test['palabra'].fillna('').astype(str)
    print(f"Test set: {len(df_test)} ejemplos | {df_test['y_true'].sum()} cubanismos")

    # Retomar desde checkpoint si existe
    inicio = 0
    predicciones = []
    respuestas_raw = []

    if os.path.exists(CKPT_JSON):
        with open(CKPT_JSON) as f:
            ckpt = json.load(f)
        inicio = ckpt['ultimo_indice'] + 1
        predicciones = ckpt['predicciones']
        respuestas_raw = ckpt['respuestas_raw']
        print(f"Reanudando desde ejemplo {inicio} (checkpoint encontrado)")

    tokenizer, model = cargar_modelo()

    # Inferencia ejemplo a ejemplo
    total = len(df_test)
    print(f"\nIniciando inferencia ({total - inicio} ejemplos restantes)...")
    print(f"Tiempo estimado: ~{(total - inicio) * 2 // 60} minutos\n")

    for i in range(inicio, total):
        row = df_test.iloc[i]
        pred, raw = inferir(row['oracion'], row['palabra'], tokenizer, model)
        predicciones.append(pred)
        respuestas_raw.append(raw)

        # Progreso
        if (i + 1) % 10 == 0 or i == total - 1:
            n_done = i + 1
            n_pos  = sum(predicciones)
            print(f"  [{n_done:>3}/{total}] Predijo 1 hasta ahora: {n_pos} "
                  f"| Respuesta raw: '{raw.strip()[:20]}'")

        # Checkpoint parcial
        if (i + 1) % CHECKPOINT_N == 0:
            ckpt_data = {
                'ultimo_indice': i,
                'predicciones': predicciones,
                'respuestas_raw': respuestas_raw,
            }
            with open(CKPT_JSON, 'w') as f:
                json.dump(ckpt_data, f)
            print(f"  [CHECKPOINT guardado en ejemplo {i}]")

    # ── Métricas finales ──────────────────────────────────────────────────────
    y_true = df_test['y_true'].astype(int).tolist()
    y_pred = predicciones

    p, r, f1_cub, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print("RESULTADO FINAL — TEST SET SAGRADO")
    print(f"{'='*60}")
    print(f"  Precision : {p:.4f} | Recall: {r:.4f}")
    print(f"  F1 (CUB)  : {f1_cub:.4f} | F1-macro: {f1_mac:.4f} | MCC: {mcc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Guardar predicciones ──────────────────────────────────────────────────
    df_out = df_test.copy()
    df_out['pred_cecilia'] = y_pred
    df_out['respuesta_raw'] = respuestas_raw
    df_out.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')
    print(f"\nPredicciones guardadas en: {OUT_CSV}")

    # ── Tabla comparativa final ───────────────────────────────────────────────
    resultado_cecilia = {
        "Modelo":        f"CecilIA 2B (zero-shot, corpus cubano)",
        "Precision_CUB": round(float(p), 4),
        "Recall_CUB":    round(float(r), 4),
        "F1_CUB":        round(float(f1_cub), 4),
        "F1_macro":      round(float(f1_mac), 4),
        "MCC":           round(float(mcc), 4),
    }

    df_comp = pd.DataFrame(RESULTADOS_REFERENCIA + [resultado_cecilia])
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA — ZERO-SHOT vs FINE-TUNED")
    print(df_comp[['Modelo','F1_CUB','F1_macro','MCC']].to_string(index=False))

    comp_path = os.path.join(BASE_DIR, "comparativa_fase6_zeroshot.csv")
    df_comp.to_csv(comp_path, sep=';', index=False)
    print(f"\nTabla guardada en: {comp_path}")

    # Limpiar checkpoint si todo terminó bien
    if os.path.exists(CKPT_JSON):
        os.remove(CKPT_JSON)


if __name__ == "__main__":
    main()
