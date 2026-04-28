"""
train_multi.py - v3.1 (Token Classification / NER)
Fase 5 - El Estado del Arte (Comparativa de Transformers)

Tarea: Token Classification con esquema BIO.
- 0 (O):     Palabra normal
- 1 (B-CUB): Inicio de cubanismo
- 2 (I-CUB): Continuación de locución cubana
"""

import os
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# --- CONFIGURACIÓN DE MODELOS (Ampliada para la Tesis) ---
MODELS = {
    "MarIA_Clasico": "BSC-TeMU/roberta-base-bne",
    "BETO":  "dccuchile/bert-base-spanish-wwm-cased",
    "BERTIN": "bertin-project/bertin-roberta-base-spanish",
    "mRoBERTa_Moderno": "BSC-LT/mRoBERTa"
}

LABEL_LIST = ["O", "B-CUB", "I-CUB"]
NUM_LABELS = 3

# Ruta flexible
POSIBLES_RUTAS = [
    "/content/dataset_bio_cubanismos",
    "dataset_bio_cubanismos",
]
DATASET_PATH = next((p for p in POSIBLES_RUTAS if os.path.exists(p)), "dataset_bio_cubanismos")

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Ignorar los tokens especiales (etiqueta = -100)
    true_predictions = [
        [LABEL_LIST[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Tokens especiales
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])  # Primera subpalabra
            else:
                label_ids.append(-100)  # Subpalabras adicionales ignoradas
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def train_model(alias, model_id, dataset):
    print(f"\n{'='*60}")
    print(f"🚀 PROCESANDO (NER): {alias} ({model_id})")
    print(f"{'='*60}")

    # 1. FIX CRÍTICO: Tokenizadores de RoBERTa requieren add_prefix_space=True para NER
    if "roberta" in model_id.lower() or "bertin" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 2. FIX CRÍTICO: Fallback de Safetensors para modelos clásicos
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            id2label={0: "O", 1: "B-CUB", 2: "I-CUB"},
            label2id={"O": 0, "B-CUB": 1, "I-CUB": 2},
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ Reintentando {alias} sin safetensors debido a: {e}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            id2label={0: "O", 1: "B-CUB", 2: "I-CUB"},
            label2id={"O": 0, "B-CUB": 1, "I-CUB": 2},
            trust_remote_code=True,
            use_safetensors=False
        )

    output_dir = f"./resultados_ner_{alias.lower()}"

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # En versiones de transformers < 4.41 usar: evaluation_strategy="epoch"
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"🔥 Entrenando {alias}...")
    trainer.train()
    metrics = trainer.evaluate()

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics

def main():
    print("📦 Cargando dataset BIO NER...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(dataset)
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return

    resultados = []

    for alias, model_id in MODELS.items():
        try:
            m = train_model(alias, model_id, dataset)
            resultados.append({
                "Modelo": alias,
                "Precision": round(m["eval_precision"], 4),
                "Recall": round(m["eval_recall"], 4),
                "F1-Score": round(m["eval_f1"], 4),
                "Accuracy": round(m["eval_accuracy"], 4),
            })
        except Exception as e:
            print(f"❌ ERROR CRÍTICO EN {alias}: {e}")
            resultados.append({"Modelo": alias, "Precision": 0, "Recall": 0, "F1-Score": 0, "Accuracy": 0})

    df_res = pd.DataFrame(resultados)
    df_res.to_csv("comparativa_final_ner.csv", index=False, sep=";")

    print("\n" + "="*60)
    print("🏆 COMPARATIVA FINAL NER (BIO)")
    print("="*60)
    print(df_res.to_string(index=False))
    print("\nArchivo guardado: comparativa_final_ner.csv")

if __name__ == "__main__":
    main()