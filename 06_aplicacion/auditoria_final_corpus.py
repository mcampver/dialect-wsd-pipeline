"""
AUDITORÍA FINAL DEL CORPUS (15,000 REGISTROS)
=============================================
Este script procesa el corpus masivo usando el modelo BETO fine-tuned
y genera una comparativa entre el criterio humano, la IA previa y BETO.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_cubanismos_final")
INPUT_CSV = os.path.join(BASE_DIR, "..", "03_datos", "corpus_preetiquetado.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "04_resultados", "auditoria_corpus_completo.csv")

def audit_corpus():
    print("--- Iniciando Auditoría Masiva con BETO ---")
    
    # 1. Cargar Modelo y Tokenizer
    print("Cargando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Usar GPU si está disponible, si no CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Usando dispositivo: {device}")

    # 2. Leer Corpus
    print(f"Leyendo archivo: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, sep=';', encoding='utf-8')
    
    # Asegurar que las columnas existen
    if 'oracion' not in df.columns or 'cubanismo_detectado' not in df.columns:
        print("Error: El CSV no tiene las columnas 'oracion' o 'cubanismo_detectado'")
        return

    # 3. Procesamiento por Lotes (Batch) para velocidad
    batch_size = 16 
    results = []
    
    print(f"Procesando {len(df)} registros...")
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        
        sentences = batch_df['oracion'].tolist()
        words = batch_df['cubanismo_detectado'].tolist()
        
        # Tokenización del batch
        inputs = tokenizer(
            sentences, 
            text_pair=words, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Aplicar Softmax para obtener probabilidades
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # La clase 1 es "Cubanismo"
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
            
        results.extend(predictions)

    # 4. Guardar resultados
    df['ia_beto_final'] = results
    
    # Crear carpeta de resultados si no existe
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, sep=';', index=False, encoding='utf-8')
    print(f"\n✅ Auditoría completada. Guardado en: {OUTPUT_CSV}")

    # 5. Generar reporte estadístico rápido
    print("\n" + "="*40)
    print("RESUMEN DE RESULTADOS (BETO vs Otros)")
    print("="*40)
    
    # Acuerdo con Humano (es_cubanismo)
    # Convertimos a int por si acaso
    df['es_cubanismo'] = pd.to_numeric(df['es_cubanismo'], errors='coerce').fillna(0).astype(int)
    
    acuerdo_humano = (df['es_cubanismo'] == df['ia_beto_final']).sum()
    pct_acuerdo = (acuerdo_humano / len(df)) * 100
    
    print(f"Coincidencia con Humano (es_cubanismo): {pct_acuerdo:.2f}%")
    
    # Detecciones totales
    total_ia_previa = pd.to_numeric(df['ia_sugiere'], errors='coerce').fillna(0).astype(int).sum()
    total_beto = df['ia_beto_final'].sum()
    
    print(f"Total cubanismos sugeridos por IA previa: {total_ia_previa}")
    print(f"Total cubanismos detectados por BETO: {total_beto}")
    
    if total_ia_previa > 0:
        diferencia = ((total_beto - total_ia_previa) / total_ia_previa) * 100
        print(f"Variación en detección: {diferencia:+.2f}%")

if __name__ == "__main__":
    audit_corpus()
