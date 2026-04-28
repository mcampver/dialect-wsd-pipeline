"""
comparativa_final.py
====================
Consolida los resultados de TODAS las fases del pipeline en una
única tabla comparativa científica lista para la tesis.

Requiere (en el mismo directorio):
  - comparativa_baselines.csv    (Fase 3)
  - comparativa_fase4.csv        (Fase 4)
  - comparativa_fase5.csv        (Fase 5, primera ejecución BETO+MrBERT-es)
  - comparativa_fase5 (1).csv    (Fase 5, segunda ejecución BERTIN+mDeBERTa+XLM-R)
  - comparativa_fase6_zeroshot.csv (Fase 6, opcional)

Salida:
  - tabla_comparativa_final.csv
  - tabla_comparativa_final.md   (lista para pegar en la tesis)
"""

import pandas as pd
import os

BASE_DIR = "/content" if os.path.exists("/content") else "."

# ── Fase 3: Baselines lingüísticos ────────────────────────────────────────────
baselines = [
    {"Modelo": "B0 — Clase mayoritaria",  "Fase": "F3 Reglas",
     "Precision_CUB": 0.000, "Recall_CUB": 0.000, "F1_CUB": 0.000,
     "F1_macro": 0.483, "MCC": 0.000},
    {"Modelo": "B1 — Diccionario puro",   "Fase": "F3 Reglas",
     "Precision_CUB": 0.066, "Recall_CUB": 1.000, "F1_CUB": 0.124,
     "F1_macro": 0.062, "MCC": 0.000},
    {"Modelo": "B2 — Dicc. + POS",        "Fase": "F3 Reglas",
     "Precision_CUB": 0.080, "Recall_CUB": 0.931, "F1_CUB": 0.147,
     "F1_macro": 0.266, "MCC": 0.101},
    {"Modelo": "B3 — Dicc. + POS + DEP",  "Fase": "F3 Reglas",
     "Precision_CUB": 0.091, "Recall_CUB": 0.931, "F1_CUB": 0.166,
     "F1_macro": 0.336, "MCC": 0.144},
]

# ── Fase 4: ML Clásico ────────────────────────────────────────────────────────
ml_clasico = [
    {"Modelo": "RF — Random Forest",        "Fase": "F4 ML Clásico",
     "Precision_CUB": 0.429, "Recall_CUB": 0.103, "F1_CUB": 0.167,
     "F1_macro": 0.566, "MCC": 0.186},
    {"Modelo": "GB — Gradient Boosting",    "Fase": "F4 ML Clásico",
     "Precision_CUB": 0.308, "Recall_CUB": 0.138, "F1_CUB": 0.190,
     "F1_macro": 0.575, "MCC": 0.170},
    {"Modelo": "SVM — LinearSVC",           "Fase": "F4 ML Clásico",
     "Precision_CUB": 0.179, "Recall_CUB": 0.241, "F1_CUB": 0.206,
     "F1_macro": 0.570, "MCC": 0.142},
    {"Modelo": "LR — Reg. Logística",       "Fase": "F4 ML Clásico",
     "Precision_CUB": 0.213, "Recall_CUB": 0.448, "F1_CUB": 0.289,
     "F1_macro": 0.604, "MCC": 0.238},
]

# ── Fase 5: Transformers ──────────────────────────────────────────────────────
transformers_results = [
    {"Modelo": "BERTIN (RoBERTa, mC4)",            "Fase": "F5 Transformer",
     "Precision_CUB": 0.000, "Recall_CUB": 0.000, "F1_CUB": 0.000,
     "F1_macro": 0.483, "MCC": 0.000},
    {"Modelo": "mDeBERTa-v3 (DeBERTa, multilingüe)", "Fase": "F5 Transformer",
     "Precision_CUB": 0.000, "Recall_CUB": 0.000, "F1_CUB": 0.000,
     "F1_macro": 0.483, "MCC": 0.000},
    {"Modelo": "XLM-RoBERTa (multilingüe masivo)",  "Fase": "F5 Transformer",
     "Precision_CUB": 0.159, "Recall_CUB": 0.241, "F1_CUB": 0.192,
     "F1_macro": 0.559, "MCC": 0.125},
    {"Modelo": "MrBERT-es (ModernBERT, BNE)",       "Fase": "F5 Transformer",
     "Precision_CUB": 0.368, "Recall_CUB": 0.483, "F1_CUB": 0.418,
     "F1_macro": 0.685, "MCC": 0.375},
    {"Modelo": "BETO (BERT, Wikipedia lat.)",        "Fase": "F5 Transformer",
     "Precision_CUB": 0.390, "Recall_CUB": 0.552, "F1_CUB": 0.457,
     "F1_macro": 0.705, "MCC": 0.419},
]

# ── Fase 6: Zero-Shot LLM ─────────────────────────────────────────────────────
# Todos los modelos LLM se cargan dinámicamente desde sus CSVs.
# No hay resultados hardcoded para evitar errores de transcripción.
zeroshot = []


def _cargar_llm_csv(path: str, alias: str) -> dict | None:
    """Carga un resultado LLM desde su CSV de comparativa."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=';')
    row = df[df['Modelo'].str.contains(alias, case=False)]
    if row.empty:
        row = df.iloc[[0]]   # tomar primera fila si no hay match por nombre
    r = row.iloc[0]
    return {
        "Modelo":        r['Modelo'],
        "Fase":          "F6 LLM Zero-Shot",
        "Precision_CUB": r['Precision_CUB'],
        "Recall_CUB":    r['Recall_CUB'],
        "F1_CUB":        r['F1_CUB'],
        "F1_macro":      r['F1_macro'],
        "MCC":           r['MCC'],
    }


def main():
    print("=" * 65)
    print("TABLA COMPARATIVA FINAL — PIPELINE COMPLETO")
    print("=" * 65)

    filas = baselines + ml_clasico + transformers_results + zeroshot

    # Qwen2.5-2B (Fase 6b) — cargado desde CSV si existe
    qwen_path = os.path.join(BASE_DIR, "comparativa_fase6b_qwen.csv")
    qwen = _cargar_llm_csv(qwen_path, "Qwen")
    if qwen:
        filas.append(qwen)
        print(f"Qwen incluido: {qwen['Modelo']}")
    else:
        filas.append({
            "Modelo": "Qwen2.5-2B-Instruct (zero-shot, genérico)", "Fase": "F6 LLM Zero-Shot",
            "Precision_CUB": None, "Recall_CUB": None, "F1_CUB": None,
            "F1_macro": None, "MCC": None,
        })
        print("AVISO: Qwen aún no ejecutado (Fase 6b pendiente)")

    # CecilIA 2B (Fase 6) — cargado desde CSV si existe
    cecilia_path = os.path.join(BASE_DIR, "comparativa_fase6_zeroshot.csv")
    cecilia = _cargar_llm_csv(cecilia_path, "CecilIA")
    if cecilia:
        filas.append(cecilia)
        print(f"CecilIA incluida: {cecilia['Modelo']}")
    else:
        filas.append({
            "Modelo": "CecilIA 2B (zero-shot, corpus cubano)", "Fase": "F6 LLM Zero-Shot",
            "Precision_CUB": None, "Recall_CUB": None, "F1_CUB": None,
            "F1_macro": None, "MCC": None,
        })
        print("AVISO: CecilIA aún no ejecutada (Fase 6 pendiente)")

    # CecilIA-Instruct QLoRA (Fase 7)
    qlora_path = os.path.join(BASE_DIR, "comparativa_fase7_cecilia_qlora.csv")
    qlora = _cargar_llm_csv(qlora_path, "CecilIA")
    if qlora:
        qlora["Fase"] = "F7 CecilIA QLoRA"
        filas.append(qlora)
        print(f"CecilIA QLoRA incluida: {qlora['Modelo']}")
    else:
        filas.append({
            "Modelo": "CecilIA-Instruct 2B (QLoRA, corpus cubano)", "Fase": "F7 CecilIA QLoRA",
            "Precision_CUB": None, "Recall_CUB": None, "F1_CUB": None,
            "F1_macro": None, "MCC": None,
        })
        print("AVISO: CecilIA QLoRA aún no ejecutada (Fase 7 pendiente)")

    df = pd.DataFrame(filas)

    # ── Tabla en consola ──────────────────────────────────────────────────────
    print(f"\n{'Modelo':<42} {'Fase':<18} {'F1_CUB':>7} {'MCC':>7}")
    print("-" * 78)
    fase_actual = None
    for _, r in df.iterrows():
        if r['Fase'] != fase_actual:
            if fase_actual is not None:
                print()
            fase_actual = r['Fase']
        f1  = f"{r['F1_CUB']:.3f}" if r['F1_CUB'] is not None else "  ---"
        mcc = f"{r['MCC']:.3f}"    if r['MCC']    is not None else "  ---"
        print(f"  {r['Modelo']:<40} {r['Fase']:<18} {f1:>7} {mcc:>7}")

    # ── Mejor modelo por fase ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MEJOR MODELO POR FASE (F1 CUB)")
    print(f"{'='*65}")
    for fase in df['Fase'].unique():
        sub = df[df['Fase'] == fase].dropna(subset=['F1_CUB'])
        if sub.empty:
            continue
        best = sub.loc[sub['F1_CUB'].idxmax()]
        print(f"  {fase:<20} → {best['Modelo']:<38} F1={best['F1_CUB']:.3f}  MCC={best['MCC']:.3f}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(BASE_DIR, "tabla_comparativa_final.csv")
    df.to_csv(csv_path, sep=';', index=False)
    print(f"\nCSV guardado en: {csv_path}")

    # ── Markdown para la tesis ────────────────────────────────────────────────
    md_lines = [
        "| Modelo | Fase | Precision | Recall | F1 (CUB) | F1-macro | MCC |",
        "|--------|------|-----------|--------|-----------|----------|-----|",
    ]
    for _, r in df.iterrows():
        def fmt(v): return f"{v:.3f}" if v is not None else "---"
        md_lines.append(
            f"| {r['Modelo']} | {r['Fase']} | {fmt(r['Precision_CUB'])} | "
            f"{fmt(r['Recall_CUB'])} | {fmt(r['F1_CUB'])} | "
            f"{fmt(r['F1_macro'])} | {fmt(r['MCC'])} |"
        )

    md_path = os.path.join(BASE_DIR, "tabla_comparativa_final.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Tabla Comparativa Final — Pipeline Completo\n\n")
        f.write("> Test set: 438 ejemplos (29 cubanismos, 409 no cubanismos). "
                "Todas las fases evaluadas sobre el mismo conjunto.\n\n")
        f.write('\n'.join(md_lines))
    print(f"Markdown guardado en: {md_path}")


if __name__ == "__main__":
    main()
