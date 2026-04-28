"""
crear_split_canonico.py
=======================
EJECUTAR SOLO UNA VEZ. Congela el split Train/Test que comparten
TODAS las fases del pipeline.

  split_train.csv  → 80% estratificado (para entrenar y hacer CV)
  split_test.csv   → 20% estratificado (TEST SAGRADO — solo para reporte final)

El test set nunca se toca durante el entrenamiento de ningún modelo.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuración fija (NO cambiar) ───────────────────────────────────────────
INPUT_CSV    = "resultados_baseline.csv"
TRAIN_CSV    = "split_train.csv"
TEST_CSV     = "split_test.csv"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("📦 CREANDO SPLIT CANÓNICO TRAIN/TEST")
    print("=" * 55)

    df = pd.read_csv(INPUT_CSV, sep=';', encoding='utf-8-sig')
    print(f"Total ejemplos cargados: {len(df)}")
    print(f"\nDistribución global:")
    vc = df['y_true'].value_counts()
    print(f"  Clase 0 (No cubanismo): {vc.get(0, 0):>5}  ({vc.get(0,0)/len(df)*100:.1f}%)")
    print(f"  Clase 1 (Cubanismo):    {vc.get(1, 0):>5}  ({vc.get(1,0)/len(df)*100:.1f}%)")

    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['y_true']
    )

    df_train.to_csv(TRAIN_CSV, sep=';', index=False, encoding='utf-8-sig')
    df_test.to_csv(TEST_CSV,  sep=';', index=False, encoding='utf-8-sig')

    print(f"\n✅ TRAIN : {len(df_train):>5} ejemplos | "
          f"{df_train['y_true'].sum():>3} cubanismos ({df_train['y_true'].mean()*100:.1f}%)")
    print(f"✅ TEST  : {len(df_test):>5} ejemplos | "
          f"{df_test['y_true'].sum():>3} cubanismos ({df_test['y_true'].mean()*100:.1f}%)")
    print(f"\n💾 Guardado: {TRAIN_CSV} / {TEST_CSV}")
    print("⚠️  NO volver a ejecutar — el split debe permanecer idéntico en todas las fases.")

if __name__ == "__main__":
    main()
