# Configuración del modelo (MarIA es el recomendado por el mentor)
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne" 
DATASET_PATH = "dataset_cubanismos_hf"
OUTPUT_DIR = "modelo_cubanismos_maria"

def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_prec = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_acc = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": metric_acc.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": metric_prec.compute(predictions=predictions, references=labels, zero_division=0)["precision"],
        "recall": metric_recall.compute(predictions=predictions, references=labels, zero_division=0)["recall"],
        "f1": metric_f1.compute(predictions=predictions, references=labels, zero_division=0)["f1"],
    }

def train():
    print(f"📦 Cargando dataset desde {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    
    print(f"🔍 Cargando Tokenizer y Modelo: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Añadimos nuestro token especial de target si no existe
    # (En la práctica técnica, Roberts-BNE ya maneja bien el contexto, 
    # pero podemos añadir tokens especiales si fuera necesario)
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("🚀 Tokenizando datos...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    # model.resize_token_embeddings(len(tokenizer)) # Si añadimos tokens especiales

    # Configuración de Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5, # 5 épocas para empezar
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🔥 Iniciando Entrenamiento (Fine-Tuning)...")
    print("Nota: Esto requiere una GPU (NVIDIA con CUDA) para ser eficiente.")
    
    trainer.train()
    
    print("\n✅ Entrenamiento finalizado. Evaluando modelo final...")
    eval_results = trainer.evaluate()
    print(f"Resultados Finales: {eval_results}")
    
    # Guardar el modelo final
    trainer.save_model(OUTPUT_DIR)
    print(f"💾 Modelo guardado en: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
