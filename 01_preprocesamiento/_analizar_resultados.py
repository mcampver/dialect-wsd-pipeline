import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score

# Cargar todos los resultados individuales
archivos = {
    "BETO":         "resultados_fase5_BETO.csv",
    "MrBERT-es":    "resultados_fase5_MrBERT-es.csv",
    "BERTIN":       "resultados_fase5_BERTIN.csv",
    "mDeBERTa-v3":  "resultados_fase5_mDeBERTa-v3.csv",
    "XLM-RoBERTa":  "resultados_fase5_XLM-RoBERTa.csv",
}

for alias, fname in archivos.items():
    df = pd.read_csv(fname, sep=';', encoding='utf-8-sig')
    pred_col = [c for c in df.columns if c.startswith('pred_')][0]
    y_true = df['y_true'].astype(int).tolist()
    y_pred = df[pred_col].astype(int).tolist()
    n_pred1 = sum(y_pred)
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    print(f"{alias:15s} | Predijo 1: {n_pred1:3d} | TP={tp:2d} FP={fp:2d} FN={fn:2d}")
