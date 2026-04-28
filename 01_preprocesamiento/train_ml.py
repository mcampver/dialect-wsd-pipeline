"""
train_ml.py
Fase 4 - Experimentación y Modelos Avanzados (Machine Learning CLásico)

Este script entrena un modelo predictivo basado en Machine Learning (Random Forest)
para detectar cubanismos. A diferencia del diccionario crudo, el modelo aprende a 
desambiguar utilizando las funciones sintácticas de spaCy (Fase 3) y el contexto 
de la oración (TF-IDF).

Pasos:
1. Carga los datos de 'resultados_baseline.csv'.
2. Extrae características (Features): Texto (TF-IDF) + Sintaxis (One-Hot Encoding).
3. Entrena un clasificador Random Forest.
4. Evalúa en el conjunto de prueba (Test Set) y muestra las métricas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("==================================================")
    print("🧠 INICIANDO FASE 4: ENTRENAMIENTO DE MACHINE LEARNING")
    print("==================================================")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv('resultados_baseline.csv', sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print("Error: No se encuentra 'resultados_baseline.csv'. Ejecuta la Fase 3 primero.")
        return

    print(f"Datos cargados: {len(df)} oraciones anotadas.")
    
    # Asegurar que no hay valores nulos en texto y features
    df['oracion'] = df['oracion'].fillna('')
    df['spacy_pos'] = df['spacy_pos'].fillna('UNKNOWN')
    df['spacy_dep'] = df['spacy_dep'].fillna('UNKNOWN')
    df['cat_diccionario'] = df['cat_diccionario'].fillna('UNKNOWN')
    
    # Separar Features (X) y Etiquetas (y)
    X = df[['oracion', 'spacy_pos', 'spacy_dep', 'cat_diccionario']]
    y = df['y_true'].astype(int)

    # 2. Dividir en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nDivisión: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    # 3. Crear el Pipeline de Extracción de Características
    # Usaremos TF-IDF para capturar el contexto semántico de la oración completa
    # y OneHotEncoder para las variables categóricas sintácticas.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), 'oracion'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['spacy_pos', 'spacy_dep', 'cat_diccionario'])
        ]
    )

    # 4. Definir el Modelo (Random Forest suele manejar muy bien textos escasos + variables categóricas)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    print("\nEntrenando modelo Random Forest...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluación del modelo
    print("\nEvaluando en el conjunto de prueba (Test Set)...")
    y_pred = pipeline.predict(X_test)
    
    print("\n--- MÉTRICAS DEL MODELO DE MACHINE LEARNING ---")
    print(classification_report(y_test, y_pred, target_names=['0 (Falso Positivo)', '1 (Cubanismo Real)'], zero_division=0))
    
    print("Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Extraer las palabras para ver ejemplos en los que falló
    df_test = df.iloc[X_test.index].copy()
    df_test['prediccion_ml'] = y_pred
    
    errores = df_test[df_test['y_true'] != df_test['prediccion_ml']]
    if not errores.empty:
        print(f"\n[!] El modelo cometió {len(errores)} errores en el Test Set.")
        print("Guardando predicciones del test set en 'resultados_ml_test.csv' para análisis...")
        df_test.to_csv("resultados_ml_test.csv", sep=";", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
