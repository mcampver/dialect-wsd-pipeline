"""
fase6b_qwen_zeroshot.py
=======================
Experimento Zero-Shot con Qwen2.5-2B-Instruct — LLM genérico de referencia.

Propósito: establecer un baseline justo para comparar con CecilIA 2B (corpus cubano).
Los dos modelos tienen tamaño similar (~2B parámetros) pero corpus de preentrenamiento
radicalmente distintos:

  Qwen2.5-2B-Instruct → corpus web genérico multilingüe (no contiene datos cubanos)
  CecilIA 2B          → 1,000M tokens exclusivamente en español de Cuba (Fase 6a)

Al usar el MISMO prompt, el MISMO test set y la MISMA métrica, la diferencia en
los resultados puede atribuirse únicamente al corpus de preentrenamiento dialectal.

Requisitos Colab:
  !pip install transformers accelerate bitsandbytes scikit-learn -q
"""

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
MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
RANDOM_STATE   = 42
MAX_NEW_TOKENS = 8
CHECKPOINT_N   = 50

BASE_DIR  = "/content" if os.path.exists("/content") else "."
TEST_CSV  = os.path.join(BASE_DIR, "split_test.csv")
OUT_CSV   = os.path.join(BASE_DIR, "resultados_fase6b_qwen.csv")
CKPT_JSON = os.path.join(BASE_DIR, "qwen_checkpoint.json")
# ──────────────────────────────────────────────────────────────────────────────

# ── Prompt — idéntico al de CecilIA para garantizar comparabilidad ────────────
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


def parsear_respuesta(texto_generado: str) -> int:
    texto = texto_generado.strip()[:30].upper()
    texto = re.sub(r'^[^A-ZÁÉÍÓÚÑ]+', '', texto)
    if texto.startswith('S'):
        return 1
    if texto.startswith('N'):
        return 0
    if re.search(r'\bS[IÍ]\b', texto):
        return 1
    if re.search(r'\bNO\b', texto):
        return 0
    return 0


def cargar_modelo():
    print(f"Cargando modelo: {MODEL_ID}")
    print("Cuantización 4-bit (NF4) para GPU T4...")

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
    print(f"Modelo cargado (~{sum(p.numel() for p in model.parameters())/1e9:.1f}B parámetros)")
    return tokenizer, model


def inferir(oracion: str, palabra: str, tokenizer, model) -> tuple[int, str]:
    prompt  = construir_prompt(oracion, palabra)
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_input = inputs["input_ids"].shape[1]
    texto   = tokenizer.decode(output_ids[0, n_input:], skip_special_tokens=True)
    return parsear_respuesta(texto), texto


def main():
    print("=" * 60)
    print("FASE 6b: QWEN 2.5-2B ZERO-SHOT (LLM Genérico de Referencia)")
    print("=" * 60)

    df_test = pd.read_csv(TEST_CSV, sep=';', encoding='utf-8-sig')
    df_test['oracion'] = df_test['oracion'].fillna('').astype(str)
    df_test['palabra'] = df_test['palabra'].fillna('').astype(str)
    print(f"Test set: {len(df_test)} ejemplos | {df_test['y_true'].sum()} cubanismos")

    # Reanudar desde checkpoint
    inicio = 0
    predicciones, respuestas_raw = [], []
    if os.path.exists(CKPT_JSON):
        with open(CKPT_JSON) as f:
            ckpt = json.load(f)
        inicio       = ckpt['ultimo_indice'] + 1
        predicciones = ckpt['predicciones']
        respuestas_raw = ckpt['respuestas_raw']
        print(f"Reanudando desde ejemplo {inicio}")

    tokenizer, model = cargar_modelo()

    total = len(df_test)
    print(f"\nIniciando inferencia ({total - inicio} ejemplos)...\n")

    for i in range(inicio, total):
        row  = df_test.iloc[i]
        pred, raw = inferir(row['oracion'], row['palabra'], tokenizer, model)
        predicciones.append(pred)
        respuestas_raw.append(raw)

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{i+1:>3}/{total}] Predijo 1: {sum(predicciones):>3} "
                  f"| Raw: '{raw.strip()[:20]}'")

        if (i + 1) % CHECKPOINT_N == 0:
            with open(CKPT_JSON, 'w') as f:
                json.dump({'ultimo_indice': i, 'predicciones': predicciones,
                           'respuestas_raw': respuestas_raw}, f)
            print(f"  [CHECKPOINT guardado en ejemplo {i}]")

    # ── Métricas ──────────────────────────────────────────────────────────────
    y_true = df_test['y_true'].astype(int).tolist()
    y_pred = predicciones

    p, r, f1_cub, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print("RESULTADO FINAL — TEST SET SAGRADO")
    print(f"{'='*60}")
    print(f"  Precision : {p:.4f} | Recall: {r:.4f}")
    print(f"  F1 (CUB)  : {f1_cub:.4f} | F1-macro: {f1_mac:.4f} | MCC: {mcc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    df_out = df_test.copy()
    df_out['pred_qwen'] = y_pred
    df_out['respuesta_raw'] = respuestas_raw
    df_out.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')

    res_csv = os.path.join(BASE_DIR, "comparativa_fase6b_qwen.csv")
    pd.DataFrame([{
        "Modelo":        "Qwen2.5-2B-Instruct (zero-shot, genérico)",
        "Precision_CUB": round(float(p), 4),
        "Recall_CUB":    round(float(r), 4),
        "F1_CUB":        round(float(f1_cub), 4),
        "F1_macro":      round(float(f1_mac), 4),
        "MCC":           round(float(mcc), 4),
    }]).to_csv(res_csv, sep=';', index=False)

    print(f"\nResultados guardados en: {OUT_CSV}")
    print(f"CSV resumen en: {res_csv}")

    # Limpiar checkpoint
    if os.path.exists(CKPT_JSON):
        os.remove(CKPT_JSON)


if __name__ == "__main__":
    main()
