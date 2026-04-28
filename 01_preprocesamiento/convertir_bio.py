"""
convertir_bio.py
Fase 5 - Conversión del Ground Truth al Formato BIO (NER)

Transforma el CSV de ground truth (una oración + una palabra objetivo) 
al formato de listas de tokens con etiquetas BIO requerido para 
Token Classification en Hugging Face.

Esquema de etiquetas:
  0 = O       (Outside - Palabra normal)
  1 = B-CUB   (Begin - Inicio de cubanismo)
  2 = I-CUB   (Inside - Continuación de locución)
"""

import pandas as pd
from datasets import Dataset, DatasetDict, Sequence, ClassLabel, Features
from sklearn.model_selection import train_test_split

LABEL2ID = {"O": 0, "B-CUB": 1, "I-CUB": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def oracion_a_bio(oracion: str, palabra_objetivo: str, es_cubanismo: int):
    """
    Tokeniza la oración a nivel de espacios y etiqueta con BIO.
    Si es_cubanismo == 1, la palabra objetivo recibe la etiqueta B-CUB.
    Las palabras de la misma locución de más de un token reciben I-CUB.
    """
    tokens = oracion.strip().split()
    
    # Las palabras objetivo pueden ser locuciones (varias palabras)
    palabras_objetivo = str(palabra_objetivo).strip().lower().split()
    n_objetivo = len(palabras_objetivo)
    
    etiquetas = []
    i = 0
    while i < len(tokens):
        token_limpio = tokens[i].strip(".,;:!?¡¿\"'()").lower()
        
        if es_cubanismo == 1 and token_limpio == palabras_objetivo[0]:
            # Verificar si el inicio de una locución coincide
            ventana = [t.strip(".,;:!?¡¿\"'()").lower() for t in tokens[i:i+n_objetivo]]
            if ventana == palabras_objetivo:
                etiquetas.append(LABEL2ID["B-CUB"])
                for _ in range(n_objetivo - 1):
                    i += 1
                    etiquetas.append(LABEL2ID["I-CUB"])
            else:
                etiquetas.append(LABEL2ID["O"])
        else:
            etiquetas.append(LABEL2ID["O"])
        i += 1
    
    return tokens, etiquetas


def main():
    print("================================================")
    print("🏷️  CONVIRTIENDO DATASET A FORMATO BIO (NER)")
    print("================================================")
    
    # Cargar el ground truth
    try:
        df = pd.read_csv('resultados_baseline.csv', sep=';', encoding='utf-8-sig')
        print(f"Cargadas {len(df)} filas del ground truth.")
    except FileNotFoundError:
        print("Error: No se encuentra 'resultados_baseline.csv'.")
        return

    # Solo nos quedamos con el ground truth validado
    df = df.dropna(subset=['y_true', 'oracion', 'palabra'])
    df['y_true'] = df['y_true'].astype(int)

    lista_tokens = []
    lista_etiquetas = []
    omitidas = 0

    for _, row in df.iterrows():
        oracion = str(row['oracion'])
        palabra = str(row['palabra'])
        es_cub = int(row['y_true'])

        tokens, etiquetas = oracion_a_bio(oracion, palabra, es_cub)

        # Verificar alineación
        if len(tokens) != len(etiquetas):
            omitidas += 1
            continue
        
        # Sanity check: si es cubanismo, debe haber al menos un B-CUB
        if es_cub == 1 and 1 not in etiquetas:
            # La palabra no se encontró en la oración (caso edge)
            omitidas += 1
            continue

        lista_tokens.append(tokens)
        lista_etiquetas.append(etiquetas)

    print(f"✅ Ejemplos BIO generados: {len(lista_tokens)}")
    print(f"⚠️  Omitidos (no alinearon): {omitidas}")
    
    # Distribución de etiquetas
    cubanismos = sum(1 for e in lista_etiquetas if 1 in e)
    print(f"🔍 Con B-CUB (cubanismos reales): {cubanismos}")
    print(f"🔍 Sin B-CUB (no cubanismos):     {len(lista_tokens)-cubanismos}")
    
    # Construir DataFrame para split
    df_bio = pd.DataFrame({'tokens': lista_tokens, 'ner_tags': lista_etiquetas})
    
    # Split 80/20 estratificado (estratificado por si hay cubanismo o no)
    tiene_cubanism = df_bio['ner_tags'].apply(lambda x: 1 if 1 in x else 0)
    df_train, df_test = train_test_split(df_bio, test_size=0.20, random_state=42, stratify=tiene_cubanism)
    print(f"\nTrain: {len(df_train)} | Test: {len(df_test)}")
    
    # Convertir a Dataset de Hugging Face
    features = Features({
        'tokens': Sequence(feature={'dtype': 'string', '_type': 'Value'}),
        'ner_tags': Sequence(ClassLabel(names=list(LABEL2ID.keys())))
    })
    
    hf_train = Dataset.from_pandas(df_train, preserve_index=False)
    hf_test = Dataset.from_pandas(df_test, preserve_index=False)
    
    dataset = DatasetDict({'train': hf_train, 'test': hf_test})
    dataset.save_to_disk('dataset_bio_cubanismos')
    
    print("\n💾 Dataset BIO guardado en: ./dataset_bio_cubanismos/")
    print("Ya puedes lanzar train_multi.py")
    print(dataset)

if __name__ == "__main__":
    main()
