"""
fase3_baselines.py
==================
Establece los 4 baselines lingüísticos obligatorios para la comparativa científica.
Evalúa SOLO sobre split_test.csv (el test set sagrado compartido).

Baselines:
  B0 — Clase mayoritaria (predice siempre 0)   → suelo absoluto
  B1 — Diccionario puro  (predice siempre 1)   → upper bound de Recall
  B2 — Diccionario + filtro POS (spaCy)        → aporte sintáctico
  B3 — Diccionario + POS + filtro DEP          → aporte de la función sintáctica

Métricas reportadas (NO se usa accuracy — el desbalance 93/7 la hace inútil):
  Precision, Recall, F1 (clase 1), F1-macro, MCC
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ── Configuración ─────────────────────────────────────────────────────────────
TEST_CSV  = "split_test.csv"
TRAIN_CSV = "split_train.csv"
OUT_CSV   = "resultados_fase3.csv"
# ──────────────────────────────────────────────────────────────────────────────

# Mapeo categoría del diccionario → POS tags de spaCy
POS_MAP = {
    'm':    ['NOUN', 'PROPN'],
    'f':    ['NOUN', 'PROPN'],
    'm/f':  ['NOUN', 'PROPN'],
    'sust': ['NOUN', 'PROPN'],
    'v':    ['VERB', 'AUX'],
    'adj':  ['ADJ'],
    'adv':  ['ADV'],
    'prep': ['ADP'],
    'pron': ['PRON'],
    'interj': ['INTJ'],
    'conj': ['CCONJ', 'SCONJ'],
}

# Dependencias sintácticas esperadas por POS (para B3)
DEP_VALID = {
    'NOUN':  {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'nmod', 'conj', 'appos', 'root'},
    'PROPN': {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'nmod', 'conj', 'appos', 'root'},
    'VERB':  {'root', 'xcomp', 'ccomp', 'advcl', 'conj', 'acl', 'acl:relcl', 'parataxis'},
    'AUX':   {'root', 'xcomp', 'ccomp', 'advcl'},
    'ADJ':   {'amod', 'xcomp', 'conj', 'root'},
    'ADV':   {'advmod', 'obl', 'conj', 'root'},
    'INTJ':  {'discourse', 'root', 'parataxis'},
    'ADP':   {'case', 'mark'},
    'PRON':  {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'nmod'},
    'CCONJ': {'cc'},
    'SCONJ': {'mark'},
}


def get_allowed_pos(cat_dicc: str):
    """Devuelve la lista de POS válidos para una categoría del diccionario."""
    cat = str(cat_dicc).lower().strip()
    for key, pos_list in POS_MAP.items():
        if key in cat:
            return pos_list
    return None


def predecir_b2(row) -> int:
    """B2: predice 1 si el POS contextual coincide con la categoría del diccionario."""
    allowed = get_allowed_pos(row['cat_diccionario'])
    if allowed and row['spacy_pos'] in allowed:
        return 1
    return 0


def predecir_b3(row) -> int:
    """B3: B2 + filtra por función de dependencia sintáctica esperada."""
    if predecir_b2(row) == 0:
        return 0
    pos = row['spacy_pos']
    dep = str(row['spacy_dep']).lower()
    valid_deps = DEP_VALID.get(pos, None)
    if valid_deps and dep not in valid_deps:
        return 0
    return 1


def metricas(y_true, y_pred, nombre: str) -> dict:
    """Calcula y muestra el conjunto completo de métricas."""
    p, r, f1_cub, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {nombre}")
    print(f"{'─'*55}")
    print(f"  Precision (CUB) : {p:.3f}")
    print(f"  Recall    (CUB) : {r:.3f}")
    print(f"  F1        (CUB) : {f1_cub:.3f}  ← métrica principal")
    print(f"  F1-macro        : {f1_mac:.3f}")
    print(f"  MCC             : {mcc:.3f}")
    print(f"  Matriz de confusión:")
    print(f"    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")

    return {
        'Modelo': nombre,
        'Precision_CUB': round(p, 4),
        'Recall_CUB': round(r, 4),
        'F1_CUB': round(f1_cub, 4),
        'F1_macro': round(f1_mac, 4),
        'MCC': round(mcc, 4),
        'TP': int(cm[1,1]), 'FP': int(cm[0,1]),
        'TN': int(cm[0,0]), 'FN': int(cm[1,0]),
    }


def main():
    print("=" * 55)
    print("📊 FASE 3: BASELINES LINGÜÍSTICOS")
    print("=" * 55)

    # Cargar sets
    df_train = pd.read_csv(TRAIN_CSV, sep=';', encoding='utf-8-sig')
    df_test  = pd.read_csv(TEST_CSV,  sep=';', encoding='utf-8-sig')

    print(f"\nTrain: {len(df_train)} ejemplos | {df_train['y_true'].sum()} cubanismos")
    print(f"Test : {len(df_test)} ejemplos  | {df_test['y_true'].sum()} cubanismos")

    y_true = df_test['y_true'].astype(int).tolist()

    resultados = []

    # B0 — Clase mayoritaria (suelo absoluto)
    y_b0 = [0] * len(df_test)
    resultados.append(metricas(y_true, y_b0, "B0 — Clase Mayoritaria (siempre 0)"))

    # B1 — Diccionario puro (todos los candidatos = cubanismo)
    y_b1 = [1] * len(df_test)
    resultados.append(metricas(y_true, y_b1, "B1 — Diccionario Puro (siempre 1)"))

    # B2 — Filtro POS
    df_test['y_pred_b2'] = df_test.apply(predecir_b2, axis=1)
    y_b2 = df_test['y_pred_b2'].tolist()
    resultados.append(metricas(y_true, y_b2, "B2 — Diccionario + Filtro POS"))

    # B3 — Filtro POS + DEP
    df_test['y_pred_b3'] = df_test.apply(predecir_b3, axis=1)
    y_b3 = df_test['y_pred_b3'].tolist()
    resultados.append(metricas(y_true, y_b3, "B3 — Diccionario + POS + Dependencia"))

    # Tabla resumen
    df_res = pd.DataFrame(resultados)
    print(f"\n{'='*55}")
    print("📋 TABLA RESUMEN — BASELINES")
    print(df_res[['Modelo','Precision_CUB','Recall_CUB','F1_CUB','F1_macro','MCC']].to_string(index=False))

    # Guardar predicciones del test set
    df_test['y_true'] = df_test['y_true'].astype(int)
    df_test['y_pred_b0'] = 0
    df_test['y_pred_b1'] = 1
    df_test.to_csv(OUT_CSV, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n💾 Predicciones guardadas en: {OUT_CSV}")

    # Guardar tabla comparativa
    df_res.to_csv("comparativa_baselines.csv", sep=';', index=False)
    print("💾 Tabla guardada en: comparativa_baselines.csv")


if __name__ == "__main__":
    main()
