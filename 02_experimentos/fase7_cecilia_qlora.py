"""
fase7_cecilia_qlora.py
======================
Fase 7: CecilIA-Instruct + QLoRA (Parameter-Efficient Fine-Tuning)

Hipótesis: El preentrenamiento cubano (CecilIA) + supervisión contextual (QLoRA)
supera a BETO fine-tuned (F1=0.457), ya que combina:
  - Conocimiento léxico cubano (demostrado en Fase 6: Recall=0.931 zero-shot)
  - Discriminación contextual aprendida (lo que zero-shot no logró: Precision=0.068)

Colab T4 (16 GB VRAM):
  - Modelo 4-bit: ~1.2 GB  |  LoRA adapters: ~20 MB  |  Total seguro

Instalación (versiones fijadas para evitar incompatibilidades):
  !pip install transformers>=4.41.0 peft>=0.11.0 trl>=0.8.6,<0.13.0 bitsandbytes scikit-learn datasets -q
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
    BitsAndBytesConfig, TrainingArguments,
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
        if _collator_cls is not None:
            print(f"[INFO] DataCollatorForCompletionOnlyLM importado desde: {_module}")
            break
    except ImportError:
        pass
if _collator_cls is None:
    print("[WARN] DataCollatorForCompletionOnlyLM no disponible — "
          "se calcula loss sobre todos los tokens (igualmente válido)")

from trl import SFTTrainer
from datasets import Dataset

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_ID     = "gia-uh/cecilia-2b-instruct-v1"
RANDOM_STATE = 42
MAX_SEQ_LEN  = 256
OVERSAMPLE_K = 6      # repetir ejemplos positivos K veces para compensar 93/7
NUM_EPOCHS   = 3
LR           = 2e-4   # LR estándar para LoRA

# LoRA — parámetros conservadores para 2B decoder
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "v_proj"]

BASE_DIR  = "/content" if os.path.exists("/content") else "."
TRAIN_CSV = os.path.join(BASE_DIR, "split_train.csv")
TEST_CSV  = os.path.join(BASE_DIR, "split_test.csv")
OUT_CSV   = os.path.join(BASE_DIR, "resultados_fase7_cecilia_qlora.csv")
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


def cargar_modelo_qlora(tokenizer):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        dtype=torch.bfloat16,
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
    for _, row in df.iterrows():
        prompt = format_prompt_only(row['oracion'], row['palabra'])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        n = inputs["input_ids"].shape[1]
        texto = tokenizer.decode(out[0, n:], skip_special_tokens=True)
        preds.append(parsear_respuesta(texto))
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
    print("FASE 7: CECILIA-INSTRUCT + QLoRA (PEFT)")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV, sep=';', encoding='utf-8-sig')
    df_test  = pd.read_csv(TEST_CSV,  sep=';', encoding='utf-8-sig')
    for df in [df_train, df_test]:
        df['oracion'] = df['oracion'].fillna('').astype(str)
        df['palabra'] = df['palabra'].fillna('').astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds_train, ds_val, df_val = preparar_datos(df_train)
    model = cargar_modelo_qlora(tokenizer)

    collator = None
    if _collator_cls is not None:
        try:
            response_ids = tokenizer.encode(RESPONSE_TOKEN, add_special_tokens=False)
            collator = _collator_cls(response_template=response_ids, tokenizer=tokenizer)
            print("[INFO] DataCollator activado")
        except: pass

    # 1. Configuración de Entrenamiento
    output_dir = os.path.join(BASE_DIR, "cecilia_qlora_checkpoint")
    train_params = dict(
        output_dir=output_dir, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=LR, warmup_steps=100, weight_decay=0.01,
        logging_steps=20, save_strategy="epoch", eval_strategy="no",
        fp16=False, bf16=True, gradient_checkpointing=True,
        report_to="none", seed=RANDOM_STATE,
    )

    trainer_kwargs = dict(model=model, train_dataset=ds_train)
    
    # 3. Configurar Trainer de forma robusta a cambios en TRL
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if collator is not None: trainer_kwargs["data_collator"] = collator

    def create_trainer():
        # Intento 1: SFTConfig con max_seq_length (TRL reciente)
        if _sft_config_cls is not None:
            try:
                args = _sft_config_cls(**train_params, max_seq_length=MAX_SEQ_LEN, dataset_text_field="text")
                return SFTTrainer(**trainer_kwargs, args=args)
            except TypeError:
                pass
        
        # Intento 2: TrainingArguments y parámetros en SFTTrainer (TRL antiguo)
        args = TrainingArguments(**train_params)
        try:
            return SFTTrainer(
                **trainer_kwargs,
                args=args,
                max_seq_length=MAX_SEQ_LEN,
                dataset_text_field="text"
            )
        except TypeError:
            pass
            
        # Intento 3: Sin parámetros específicos de texto (dataset pre-tokenizado o default)
        return SFTTrainer(**trainer_kwargs, args=args)

    trainer = create_trainer()
    
    print("\nIniciando entrenamiento...")
    trainer.train()

    print("\nEvaluando VAL...")
    preds_val = inferir_batch(df_val, tokenizer, model)
    metricas(df_val['y_true'].astype(int).tolist(), preds_val, "VAL")

    print("\nEvaluando TEST...")
    preds_test = inferir_batch(df_test, tokenizer, model)
    res = metricas(df_test['y_true'].astype(int).tolist(), preds_test, "TEST")

    df_out = df_test.copy()
    df_out['pred_cecilia_qlora'] = preds_test
    df_out.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')

    comp_path = os.path.join(BASE_DIR, "comparativa_fase7_cecilia_qlora.csv")
    pd.DataFrame([{
        "Modelo": "CecilIA-Instruct 2B (QLoRA, corpus cubano)",
        "Precision_CUB": res['Precision_CUB'], "Recall_CUB": res['Recall_CUB'],
        "F1_CUB": res['F1_CUB'], "F1_macro": res['F1_macro'], "MCC": res['MCC'],
    }]).to_csv(comp_path, sep=';', index=False)


if __name__ == "__main__":
    main()
