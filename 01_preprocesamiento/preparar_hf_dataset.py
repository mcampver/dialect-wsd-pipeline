"""
preparar_hf_dataset.py
Fase 5 - Preparación del Estado del Arte (Transformers e IA Generativa)

Este script transforma el Ground Truth (2,212 oraciones) y la partición exacta 
(Train/Test calculada con random_seed=42) a la estructura del ecosistema
`datasets` de Hugging Face.

Esto te deja listo para ejecutar notebooks de fine-tuning (BERT, RoBERTa, MarIA, BETO).
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def main():
    print("==================================================")
    print("📦 FASE 5: FORMATEANDO DATASET PARA HUGGING FACE")
    print("==================================================")
    
    try:
        df = pd.read_csv('resultados_baseline.csv', sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print("Error: No se encuentra 'resultados_baseline.csv'.")
        return

    # Limpieza de nulos para HF
    df['oracion'] = df['oracion'].fillna('')
    df['spacy_pos'] = df['spacy_pos'].fillna('UNKNOWN')
    
    # HF suele pedir la etiqueta numérica bajo la columna 'label', 
    # y el texto principal bajo 'text'
    df_renamed = df.rename(columns={
        'oracion': 'text',
        'y_true': 'label'
    })
    
    # Para Fine-tuning estructurado, podemos pasarle la oración
    # enriquecida con la palabra a buscar (Ej: "[TARGET] asere [TARGET]")
    # Esto ayuda a Transformers genéricos a saber qué palabra desambiguar
    textos_marcados = []
    for _, row in df_renamed.iterrows():
        oracion = row['text']
        palabra = str(row['palabra'])
        
        # Marca muy simple
        oracion_marcada = oracion.replace(palabra, f"[TGT] {palabra} [TGT]")
        textos_marcados.append(oracion_marcada)
        
    df_renamed['text'] = textos_marcados
    
    # Seleccionamos las columnas relevantes para HF
    dataset_df = df_renamed[['text', 'label', 'palabra', 'cat_diccionario', 'spacy_pos']]

    # Partición EXACTA replicada del ML
    X = dataset_df.drop('label', axis=1)
    y = dataset_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Reconstruyendo los sub-dataframes
    df_train = X_train.copy()
    df_train['label'] = y_train
    
    df_test = X_test.copy()
    df_test['label'] = y_test
    
    # Convertir a Datasets de Hugging Face
    hf_train = Dataset.from_pandas(df_train, preserve_index=False)
    hf_test = Dataset.from_pandas(df_test, preserve_index=False)
    
    hg_dataset = DatasetDict({
        'train': hf_train,
        'test': hf_test
    })
    
    # Guardar a disco en formato huggingface
    hg_dataset.save_to_disk('dataset_cubanismos_hf')
    print("✅ Dataset estructurado con éxito.")
    print(hg_dataset)
    print("\n[!] Guardado en directorio: ./dataset_cubanismos_hf/")
    print("Ya puedes exportar este disco o subirlo al Hub con: hg_dataset.push_to_hub('usuario/cubanismos')")

if __name__ == "__main__":
    main()
