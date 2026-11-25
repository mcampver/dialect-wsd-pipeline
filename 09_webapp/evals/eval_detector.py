"""Eval periódico del detector con el modelo BETO real (lento, no es gate).

Mide dos cosas sobre 03_datos/split_test.csv:
  1. F1 de BETO a nivel (oración, palabra) con el umbral de producción.
  2. Recall en textos largos: cada oración positiva se incrusta tras ~250
     tokens de relleno. `analizar_texto` (por oración) debe conservar el
     recall; el modo antiguo (texto entero truncado a 128 tokens) lo pierde.

Uso:  python evals/eval_detector.py [--n 300] [--n-largos 60]
Sale con código 1 si no se alcanzan los umbrales.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "06_aplicacion"))
import clasificador_final as cf  # noqa: E402

# Línea base medida el 2026-11-25 (n=300, random_state=42): P=0.796 R=0.736
# F1=0.765. El umbral deja margen para ruido de muestreo y detecta regresiones
# del pipeline (umbral de decisión, tokenización, carga del modelo).
UMBRAL_F1 = 0.70
# Línea base: 1.0 con analizar_texto frente a 0.0 con el modo antiguo.
UMBRAL_RECALL_LARGO = 0.90
RELLENO = "Esto es una oración normal sobre economía nacional y política. " * 20


def f1(y, p):
    tp = sum(a and b for a, b in zip(y, p))
    fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and (not b) for a, b in zip(y, p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-largos", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(RAIZ, "03_datos", "split_test.csv"), sep=";", encoding="utf-8-sig")
    df = df.dropna(subset=["oracion", "palabra", "y_true"]).sample(frac=1, random_state=42)
    muestra = df.head(args.n)

    t0 = time.time()
    det = cf.DetectorCubano()
    t_carga = time.time() - t0

    # 1) clasificación (oración, palabra)
    y, p = [], []
    for o, w, t in zip(muestra["oracion"], muestra["palabra"], muestra["y_true"]):
        y.append(int(t) == 1)
        p.append(det._probabilidades(o.strip(), [w])[0] > cf.UMBRAL)
    prec, rec, f = f1(y, p)

    # 2) textos largos: positivos que BETO acierta aislados
    positivos = df[df["y_true"] == 1].head(args.n_largos * 3)
    base, nuevo, antiguo = 0, 0, 0
    for o, w in zip(positivos["oracion"], positivos["palabra"]):
        if base >= args.n_largos:
            break
        aislada = {h["palabra"].lower() for h in det.analizar_texto(o)}
        if w.lower() not in aislada:
            continue
        base += 1
        largo = RELLENO + o
        nuevo += w.lower() in {h["palabra"].lower() for h in det.analizar_texto(largo)}
        antiguo += w.lower() in {h["palabra"].lower() for h in det.analizar_frase(largo)}

    rec_largo = nuevo / base if base else 0.0
    rec_antiguo = antiguo / base if base else 0.0
    resultado = {
        "n": len(muestra), "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f, 3),
        "largos_base": base, "recall_largo_por_oracion": round(rec_largo, 3),
        "recall_largo_modo_antiguo": round(rec_antiguo, 3),
        "segundos_carga": round(t_carga, 1), "segundos_total": round(time.time() - t0, 1),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    ok = f >= UMBRAL_F1 and rec_largo >= UMBRAL_RECALL_LARGO
    print("EVAL", "OK" if ok else "FALLA", f"(F1>={UMBRAL_F1}, recall_largo>={UMBRAL_RECALL_LARGO})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
