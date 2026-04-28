"""
fase4_ml_clasico.py
===================
Clasificación binaria WSD con ML clásico.

Estrategia de evaluación:
  - 5-Fold Stratified CV sobre split_train.csv (para seleccionar el mejor modelo)
  - Evaluación final sobre split_test.csv (test set sagrado, solo al final)

Features:
  - TF-IDF de la oración completa (contexto global)
  - TF-IDF de la ventana local ±5 tokens alrededor de la palabra objetivo
  - OneHot de: spacy_pos, spacy_dep, cat_diccionario

Modelos comparados:
  LR   — Logistic Regression (baseline fuerte en NLP)
  SVM  — LinearSVC (excelente en espacios sparse de alta dimensión)
  RF   — Random Forest (referencia de la tesis anterior)
  GB   — Gradient Boosting (más potente para features mixtos)

Métricas: F1 (clase 1), F1-macro, MCC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    make_scorer, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix,
)

# ── Configuración ─────────────────────────────────────────────────────────────
TRAIN_CSV    = "split_train.csv"
TEST_CSV     = "split_test.csv"
OUT_CSV      = "resultados_fase4.csv"
RANDOM_STATE = 42
N_FOLDS      = 5
# ──────────────────────────────────────────────────────────────────────────────


def extraer_ventana(oracion: str, palabra: str, n: int = 5) -> str:
    """Extrae los N tokens de contexto alrededor de la palabra objetivo."""
    tokens = oracion.lower().split()
    p = palabra.lower().strip(".,;:!?¡¿\"'()")
    pos = next(
        (i for i, t in enumerate(tokens) if t.strip(".,;:!?¡¿\"'()") == p),
        None
    )
    if pos is None:
        return ""
    ventana = tokens[max(0, pos-n):pos] + tokens[pos+1:pos+n+1]
    return " ".join(ventana)


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna 'ventana_local' al DataFrame."""
    df = df.copy()
    df['ventana_local'] = df.apply(
        lambda r: extraer_ventana(str(r['oracion']), str(r['palabra'])), axis=1
    )
    for col in ['spacy_pos', 'spacy_dep', 'cat_diccionario']:
        df[col] = df[col].fillna('UNKNOWN').astype(str)
    df['oracion'] = df['oracion'].fillna('').astype(str)
    return df


def construir_preprocessor():
    return ColumnTransformer([
        ('tfidf_global', TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2), sublinear_tf=True
        ), 'oracion'),
        ('tfidf_local', TfidfVectorizer(
            max_features=500, ngram_range=(1, 2)
        ), 'ventana_local'),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['spacy_pos', 'spacy_dep', 'cat_diccionario']),
    ])


MODELOS = {
    "LR  — Regresión Logística": LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    ),
    "SVM — LinearSVC": LinearSVC(
        class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE
    ),
    "RF  — Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GB  — Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

SCORING = {
    'f1_cubanismo': make_scorer(f1_score, pos_label=1, zero_division=0),
    'f1_macro':     make_scorer(f1_score, average='macro', zero_division=0),
    'mcc':          make_scorer(matthews_corrcoef),
}


def evaluar_test(nombre, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    p, r, f1_cub, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average='binary', zero_division=0
    )
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc    = matthews_corrcoef(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  [TEST FINAL] {nombre}")
    print(f"  Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1_cub:.3f} | MCC: {mcc:.3f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    return {
        'Modelo': nombre.strip(),
        'Precision_CUB': round(p, 4),
        'Recall_CUB':    round(r, 4),
        'F1_CUB':        round(f1_cub, 4),
        'F1_macro':      round(f1_mac, 4),
        'MCC':           round(mcc, 4),
        'y_pred':        y_pred.tolist(),
    }


def main():
    print("=" * 60)
    print("🧠 FASE 4: ML CLÁSICO — WSD BINARIA")
    print("=" * 60)

    df_train = preparar_features(pd.read_csv(TRAIN_CSV, sep=';', encoding='utf-8-sig'))
    df_test  = preparar_features(pd.read_csv(TEST_CSV,  sep=';', encoding='utf-8-sig'))

    X_train = df_train[['oracion', 'ventana_local', 'spacy_pos', 'spacy_dep', 'cat_diccionario']]
    y_train = df_train['y_true'].astype(int)
    X_test  = df_test[['oracion', 'ventana_local', 'spacy_pos', 'spacy_dep', 'cat_diccionario']]
    y_test  = df_test['y_true'].astype(int)

    print(f"\nTrain: {len(X_train)} | Cubanismos train: {y_train.sum()}")
    print(f"Test : {len(X_test)}  | Cubanismos test : {y_test.sum()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    resultados_test = []
    mejores_pipelines = {}

    for nombre, clf in MODELOS.items():
        print(f"\n{'─'*60}")
        print(f"Modelo: {nombre}")

        pipeline = Pipeline([
            ('prep', construir_preprocessor()),
            ('clf',  clf),
        ])

        # 5-Fold CV sobre el train set
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=skf, scoring=SCORING, n_jobs=-1, error_score='raise'
        )

        f1_cv  = cv_res['test_f1_cubanismo'].mean()
        mac_cv = cv_res['test_f1_macro'].mean()
        mcc_cv = cv_res['test_mcc'].mean()
        print(f"  CV (5-fold) → F1_CUB: {f1_cv:.3f} ± {cv_res['test_f1_cubanismo'].std():.3f} | "
              f"F1-macro: {mac_cv:.3f} | MCC: {mcc_cv:.3f}")

        # Entrenar sobre todo el train set y evaluar en test
        pipeline.fit(X_train, y_train)
        mejores_pipelines[nombre] = pipeline
        res = evaluar_test(nombre, pipeline, X_test, y_test)
        res['F1_CV_mean'] = round(f1_cv, 4)
        res['F1_CV_std']  = round(cv_res['test_f1_cubanismo'].std(), 4)
        resultados_test.append(res)

    # Tabla comparativa
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred'}
                           for r in resultados_test])
    print(f"\n{'='*60}")
    print("📋 TABLA RESUMEN — ML CLÁSICO (evaluación en TEST SET)")
    print(df_res[['Modelo','F1_CUB','F1_macro','MCC','F1_CV_mean','F1_CV_std']].to_string(index=False))

    # Guardar predicciones en el test set
    df_test_out = df_test.copy()
    for res in resultados_test:
        col = "pred_" + res['Modelo'].split('—')[0].strip().replace(' ', '_')
        df_test_out[col] = res['y_pred']
    df_test_out.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n💾 Predicciones guardadas en: {OUT_CSV}")

    df_res.to_csv("comparativa_fase4.csv", sep=';', index=False)
    print("💾 Tabla guardada en: comparativa_fase4.csv")


if __name__ == "__main__":
    main()
