"""
fase7_cecilia_lora_mac.py
=========================
Fase 7: CecilIA-Instruct + LoRA (Optimizado para Mac M2 Ultra / MPS)

Diferencias con la versión de Colab:
  - NO usa BitsAndBytes (no compatible nativamente con MPS sin configuración compleja).
  - Carga el modelo en Bfloat16 directamente (la M2 Ultra tiene memoria de sobra).
  - Usa "mps" como dispositivo de aceleración.

Instalación en Mac:
  pip install transformers peft trl scikit-learn datasets pandas numpy
"""

import os, json, re, gc, inspect
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix,
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# Importación robusta de componentes de trl
_collator_cls = None
_sft_config_cls = None
try:
    from trl import SFTConfig
    _sft_config_cls = SFTConfig
except ImportError:
    pass

for _module in ["trl", "trl.trainer", "trl.trainer.utils"]:
    try:
        import importlib
        _mod = importlib.import_module(_module)
        _collator_cls = getattr(_mod, "DataCollatorForCompletionOnlyLM", None)
        if _collator_cls is not None: break
    except ImportError:
        pass

from trl import SFTTrainer
from datasets import Dataset

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_ID     = "gia-uh/cecilia-2b-instruct-v1"
RANDOM_STATE = 42
MAX_SEQ_LEN  = 256
OVERSAMPLE_K = 6      # repetir ejemplos positivos K veces para compensar 93/7
NUM_EPOCHS   = 3
LR           = 2e-4   # LR estándar para LoRA

# LoRA — parámetros conservadores
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "v_proj"]

# En local/Mac asumimos que los archivos están en el directorio actual
TRAIN_CSV = "split_train.csv"
TEST_CSV  = "split_test.csv"
OUT_CSV   = "resultados_fase7_cecilia_lora_mac.csv"
# ──────────────────────────────────────────────────────────────────────────────

RESPONSE_TOKEN = "Respuesta:"

def format_instruction(oracion: str, palabra: str, label: int) -> str:
    respuesta = "SÍ" if label == 1 else "NO"
    return (
        f'¿La palabra "{palabra}" es un cubanismo en la siguiente oración?\n'
        f'Oración: "{oracion}"\n'
        f'Responde únicamente con SÍ o NO.\n'
        f'{RESPONSE_TOKEN} {respuesta}'
    )

def format_prompt_only(oracion: str, palabra: str) -> str:
    return (
        f'¿La palabra "{palabra}" es un cubanismo en la siguiente oración?\n'
        f'Oración: "{oracion}"\n'
        f'Responde únicamente con SÍ o NO.\n'
        f'{RESPONSE_TOKEN}'
    )

def parsear_respuesta(texto: str) -> int:
    t = texto.strip()[:20].upper()
    t = re.sub(r'^[^A-ZÁÉÍÓÚÑ]+', '', t)
    if t.startswith('S'): return 1
    if t.startswith('N'): return 0
    if re.search(r'\bS[IÍ]\b', t): return 1
    return 0

def preparar_datos(df_train: pd.DataFrame):
    df_neg = df_train[df_train['y_true'] == 0]
    df_pos = df_train[df_train['y_true'] == 1]
    tr_neg, va_neg = train_test_split(df_neg, test_size=0.15, random_state=RANDOM_STATE)
    tr_pos, va_pos = train_test_split(df_pos, test_size=0.15, random_state=RANDOM_STATE)
    tr_pos_over = pd.concat([tr_pos] * OVERSAMPLE_K, ignore_index=True)
    df_tr = pd.concat([tr_neg, tr_pos_over], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    df_va = pd.concat([va_neg, va_pos], ignore_index=True)
    
    print(f"  train_inner : {len(df_tr)} ej. | {df_tr['y_true'].sum()} positivos")
    
    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        textos = [format_instruction(r['oracion'], r['palabra'], r['y_true']) for _, r in df.iterrows()]
        return Dataset.from_dict({"text": textos})

    return to_hf_dataset(df_tr), to_hf_dataset(df_va), df_va.reset_index(drop=True)

def cargar_modelo_lora(tokenizer):
    print(f"Cargando modelo en Bfloat16 (Optimizado para MPS)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", # En Mac con M2 Ultra, esto detectará MPS automáticamente
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model

def inferir_batch(df: pd.DataFrame, tokenizer, model) -> list[int]:
    model.eval()
    preds = []
    print(f"Iniciando inferencia en batch de {len(df)} ejemplos...")
    for i, row in df.iterrows():
        prompt = format_prompt_only(row['oracion'], row['palabra'])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        n = inputs["input_ids"].shape[1]
        texto = tokenizer.decode(out[0, n:], skip_special_tokens=True)
        preds.append(parsear_respuesta(texto))
        if (i+1) % 50 == 0: print(f"  Procesados: {i+1}/{len(df)}")
    return preds

def metricas(y_true, y_pred, nombre: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    print(f"\n  [{nombre}] P={p:.4f} R={r:.4f} F1={f1:.4f} MCC={mcc:.4f}")
    return {"Precision_CUB": p, "Recall_CUB": r, "F1_CUB": f1, "F1_macro": f1m, "MCC": mcc, "y_pred": y_pred}

def main():
    print("=" * 60)
    print("FASE 7: CECILIA-INSTRUCT + LORA (MAC MPS EDITION)")
    print("=" * 60)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"ERROR: No se encuentran los archivos {TRAIN_CSV} o {TEST_CSV} en el directorio actual.")
        return

    df_train = pd.read_csv(TRAIN_CSV, sep=';', encoding='utf-8-sig')
    df_test  = pd.read_csv(TEST_CSV,  sep=';', encoding='utf-8-sig')
    for df in [df_train, df_test]:
        df['oracion'] = df['oracion'].fillna('').astype(str)
        df['palabra'] = df['palabra'].fillna('').astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds_train, ds_val, df_val = preparar_datos(df_train)
    model = cargar_modelo_lora(tokenizer)

    collator = None
    if _collator_cls is not None:
        try:
            response_ids = tokenizer.encode(RESPONSE_TOKEN, add_special_tokens=False)
            collator = _collator_cls(response_template=response_ids, tokenizer=tokenizer)
            print("[INFO] DataCollator activado")
        except: pass

    # 1. Configuración de Entrenamiento
    output_dir = "./cecilia_lora_mac_checkpoint"
    train_params = dict(
        output_dir=output_dir, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=LR, warmup_steps=100, weight_decay=0.01,
        logging_steps=10, save_strategy="epoch", eval_strategy="no",
        use_mps_device=True, # Forzar uso de MPS
        bf16=True, # M2 admite bfloat16
        gradient_checkpointing=True,
        report_to="none", seed=RANDOM_STATE,
    )

    # 2. Distribución dinámica de parámetros (SFTConfig vs SFTTrainer)
    sft_params_for_config = {}
    sft_params_for_trainer = {"max_seq_length": MAX_SEQ_LEN, "dataset_text_field": "text"}

    if _sft_config_cls is not None:
        sft_config_sig = inspect.signature(_sft_config_cls.__init__)
        for p in ["max_seq_length", "dataset_text_field"]:
            if p in sft_config_sig.parameters:
                sft_params_for_config[p] = sft_params_for_trainer.pop(p)
        args = _sft_config_cls(**train_params, **sft_params_for_config)
    else:
        args = TrainingArguments(**train_params)

    # 3. Configurar Trainer
    trainer_kwargs = dict(model=model, args=args, train_dataset=ds_train, **sft_params_for_trainer)
    
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if collator is not None: trainer_kwargs["data_collator"] = collator

    trainer = SFTTrainer(**trainer_kwargs)
    
    print("\nIniciando entrenamiento en Mac...")
    trainer.train()

    print("\nEvaluando VAL...")
    preds_val = inferir_batch(df_val, tokenizer, model)
    metricas(df_val['y_true'].astype(int).tolist(), preds_val, "VAL")

    print("\nEvaluando TEST...")
    preds_test = inferir_batch(df_test, tokenizer, model)
    res = metricas(df_test['y_true'].astype(int).tolist(), preds_test, "TEST")

    df_out = df_test.copy()
    df_out['pred_cecilia_lora_mac'] = preds_test
    df_out.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')

    comp_path = "comparativa_fase7_cecilia_lora_mac.csv"
    pd.DataFrame([{
        "Modelo": "CecilIA-Instruct 2B (LoRA, Mac M2 Ultra)",
        "Precision_CUB": res['Precision_CUB'], "Recall_CUB": res['Recall_CUB'],
        "F1_CUB": res['F1_CUB'], "F1_macro": res['F1_macro'], "MCC": res['MCC'],
    }]).to_csv(comp_path, sep=';', index=False)
    print(f"\nResultados guardados en: {OUT_CSV}")


if __name__ == "__main__":
    main()
