"""
train_ml_optimized.py
Fase 4 - Experimentación y ML Tradicional (Optimizado con Undersampling)

A diferencia del script anterior, aquí atacamos el severo desbalance de clases
reduciendo los Falsos Positivos (Undersampling) en el conjunto de ENTRENAMIENTO,
obligando al modelo a no volverse "conservador" y subir así su Recall.

1. Separamos Train y Test (Test permanece intacto, mismo random_state).
2. Aplicamos RandomUnderSampler de imbalanced-learn en el pipeline.
3. Extraemos TF-IDF contextual y POS Tagging sintáctico.
4. Evaluamos y comparamos con la iteración previa.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Usamos Pipeline de imblearn para integrar el Undersampler paso a paso
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

def main():
    print("==================================================")
    print("🧠 FASE 4: ML TRADICIONAL + UNDERSAMPLING")
    print("==================================================")
    
    try:
        df = pd.read_csv('resultados_baseline.csv', sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print("Error: No se encuentra 'resultados_baseline.csv'.")
        return

    df['oracion'] = df['oracion'].fillna('')
    df['spacy_pos'] = df['spacy_pos'].fillna('UNKNOWN')
    df['spacy_dep'] = df['spacy_dep'].fillna('UNKNOWN')
    df['cat_diccionario'] = df['cat_diccionario'].fillna('UNKNOWN')
    
    X = df[['oracion', 'spacy_pos', 'spacy_dep', 'cat_diccionario']]
    y = df['y_true'].astype(int)

    # El conjunto de prueba es EXACTAMENTE el mismo (random_state=42, test_size=0.20)
    # que en train_ml.py original para garantizar comparación justa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Conjunto de Test original: {len(X_test)} muestras ({sum(y_test==1)} cubanismos)")

    # Definimos el balanceador (Undersampler)
    # Al reducir la clase mayoritaria en el training, el modelo no discrimina a la minoritaria.
    sampler = RandomUnderSampler(random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), 'oracion'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['spacy_pos', 'spacy_dep', 'cat_diccionario'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('undersampler', sampler),  # Aplica undersampling en la matriz extraída de Train
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) 
        # class_weight='balanced' ya no es necesario porque la data estará 50/50 balanceada
    ])

    print("\nEntrenando Random Forest con Undersampling...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluando en el Test Set intacto...")
    y_pred = pipeline.predict(X_test)
    
    print("\n--- NUEVAS MÉTRICAS DEL MODELO ---")
    print(classification_report(y_test, y_pred, target_names=['0 (Falso Positivo)', '1 (Cubanismo Real)'], zero_division=0))
    
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
