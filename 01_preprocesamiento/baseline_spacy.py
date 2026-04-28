"""
baseline_spacy.py
Fase 3 - Análisis y Evaluación del Modelo Base (Baseline)

Este script:
1. Lee el CSV anotado consolidando etiquetas manuales y de IA.
2. Utiliza spaCy para el análisis morfosintáctico de las oraciones.
3. Extrae Categoría Gramatical (POS) y Función Sintáctica (DEP) de la palabra.
4. Evalúa dos modelos baseline:
   - "Baseline 1 (Léxico)": Asume que todas las detecciones crudas son cubanismos (predice 1).
   - "Baseline 2 (Léxico + POS)": Usa spaCy para descartar falsos positivos si la
     categoría de la oración no coincide con la del diccionario.
5. Imprime métricas (Precision, Recall, F1, Accuracy) usando scikit-learn.
"""

import os
import csv
import glob
import spacy
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def cargar_datos_anotados():
    csv_files = glob.glob('*.csv')
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Cargando dataset: {latest_csv}")
    
    df = pd.read_csv(latest_csv, sep=';', encoding='utf-8-sig', dtype=str).fillna('')
    
    # Consolidar etiqueta ground_truth: Prioriza el manual, luego la IA
    valid_rows = []
    
    for _, row in df.iterrows():
        l_manual = row.get('es_cubanismo', '').strip()
        l_ia = row.get('ia_sugiere', '').strip()
        
        y_true = None
        if l_manual in ['0', '1']:
            y_true = int(l_manual)
        elif l_ia in ['0', '1']:
            y_true = int(l_ia)
            
        if y_true is not None:
            valid_rows.append({
                'oracion': row['oracion'],
                'palabra': row['cubanismo_detectado'],
                'cat_diccionario': row['categoria_gramatical'],
                'y_true': y_true
            })
            
    return pd.DataFrame(valid_rows)

# Mapeo simple de categorías del diccionario a POS tags de spaCy para Baseline 2
POS_MAP = {
    'm': ['NOUN', 'PROPN'],
    'f': ['NOUN', 'PROPN'],
    'm/f': ['NOUN', 'PROPN'],
    'sust': ['NOUN', 'PROPN'],
    'v': ['VERB'],
    'adj': ['ADJ'],
    'adv': ['ADV'],
    'prep': ['ADP'],
    'pron': ['PRON'],
    'interj': ['INTJ'],
    'conj': ['CCONJ', 'SCONJ']
}

def obtener_pos_spacy(cat_dicc):
    for k, v in POS_MAP.items():
        if k in cat_dicc:
            return v
    return None

def limpiar_palabra(palabra):
    return palabra.strip('.,;!?()[]{}').lower()

def procesar_spacy(df):
    print("\nCargando modelo de lenguaje de spaCy (es_core_news_md)...")
    nlp = spacy.load("es_core_news_md")
    
    print(f"Procesando {len(df)} oraciones...")
    
    y_pred_baseline1 = [] # Predice siempre 1 (Búsqueda Cruda)
    y_pred_baseline2 = [] # Predice 1 solo si coincide el POS
    
    extraidos_pos = []
    extraidos_dep = []
    
    for idx, row in df.iterrows():
        doc = nlp(row['oracion'])
        palabra_objetivo = limpiar_palabra(row['palabra'])
        
        token_match = None
        for token in doc:
            if limpiar_palabra(token.text) == palabra_objetivo:
                token_match = token
                break
                
        if token_match:
            pos = token_match.pos_
            dep = token_match.dep_
        else:
            pos = "UNKNOWN"
            dep = "UNKNOWN"
            
        extraidos_pos.append(pos)
        extraidos_dep.append(dep)
        
        # Predicción Baseline 1
        y_pred_baseline1.append(1)
        
        # Predicción Baseline 2
        allowed_pos = obtener_pos_spacy(row['cat_diccionario'])
        if allowed_pos and pos in allowed_pos:
            y_pred_baseline2.append(1)
        else:
            y_pred_baseline2.append(0) # Si el POS sintáctico no cuadra con el léxico, es falso positivo
            
    df['spacy_pos'] = extraidos_pos
    df['spacy_dep'] = extraidos_dep
    df['y_pred_b1'] = y_pred_baseline1
    df['y_pred_b2'] = y_pred_baseline2
    
    return df

def evaluar(df):
    y_true = df['y_true'].tolist()
    
    print("\n" + "="*50)
    print("📈  RESULTADOS FASE 3: EVALUACIÓN DE BASELINES")
    print("="*50)
    
    print("\n--- BASELINE 1: Extracción Cruda del Diccionario ---")
    print(" (Asumiendo que toda coincidencia es un cubanismo real)")
    y_b1 = df['y_pred_b1'].tolist()
    print(classification_report(y_true, y_b1, target_names=['0 (No Cubanismo)', '1 (Cubanismo)'], zero_division=0))
    print("Matriz de confusión:\n", confusion_matrix(y_true, y_b1))
    
    print("\n--- BASELINE 2: Funciones Sintácticas con spaCy ---")
    print(" (Predice 1 SÓLO si la categoría gramatical real del diccionario encaja en el contexto)")
    y_b2 = df['y_pred_b2'].tolist()
    print(classification_report(y_true, y_b2, target_names=['0 (No Cubanismo)', '1 (Cubanismo)'], zero_division=0))
    print("Matriz de confusión:\n", confusion_matrix(y_true, y_b2))
    
    print("\n--- FUNCIONES SINTÁCTICAS COMUNES EXTRAÍDAS ---")
    print("Top 5 dependencias sintácticas (DEP) para Positivos confirmados:")
    positivos = df[df['y_true'] == 1]
    print(positivos['spacy_dep'].value_counts().head(5).to_string())

if __name__ == "__main__":
    df_data = cargar_datos_anotados()
    if len(df_data) > 0:
        df_evaluado = procesar_spacy(df_data)
        evaluar(df_evaluado)
        df_evaluado.to_csv("resultados_baseline.csv", sep=";", index=False, encoding="utf-8-sig")
        print("\n[!] Análisis guardado en 'resultados_baseline.csv'.")
    else:
        print("No se encontraron anotaciones válidas en el CSV.")
