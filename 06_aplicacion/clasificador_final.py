"""
CLASIFICADOR FINAL DE CUBANISMOS - TESIS (VERSIÓN CORREGIDA v2.1)
==============================================================
"""

import os
import torch
import spacy
import pandas as pd
import sqlite3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuración de rutas
BASE_APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_APP_DIR, "modelo_cubanismos_final")
DB_PATH = os.path.join(BASE_APP_DIR, "..", "03_datos", "diccionario_cubanismos.db")

class DetectorCubano:
    def __init__(self):
        print("\n--- Iniciando Detector de Cubanismos (v2.1) ---")
        
        # 1. Cargar cerebro lingüístico
        try:
            self.nlp = spacy.load("es_core_news_lg")
        except:
            os.system("python -m spacy download es_core_news_lg")
            self.nlp = spacy.load("es_core_news_lg")

        # 2. Cargar IA (BETO)
        print(f"Cargando modelo BETO...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

        # 3. Cargar Diccionario de Candidatos (De la Base de Datos)
        self.candidatos = set()
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                # IMPORTANTE: La columna se llama 'lema' en tu DB
                df_cubs = pd.read_sql_query("SELECT DISTINCT lema FROM cubanismos", conn)
                self.candidatos = set(df_cubs['lema'].str.lower().unique())
                conn.close()
                print(f"Diccionario cargado: {len(self.candidatos)} términos cubanos reconocidos.")
            except Exception as e:
                print(f"Error cargando DB, usando lista de respaldo. ({e})")
        
        # Lista de respaldo con términos clave
        self.candidatos.update(["asere", "guagua", "maquina", "bola", "pinchar", "finca", "ómnibus", "asere", "que bola", "acere"])

    def analizar_frase(self, texto):
        doc = self.nlp(texto)
        hallazgos = []

        # Limpieza básica
        texto_limpio = texto.strip()

        for token in doc:
            # Usamos el LEMA (la raíz) para buscar en el diccionario
            # Así 'guaguas' -> 'guagua' y lo encuentra.
            lema_lower = token.lemma_.lower()
            palabra_original = token.text

            # PASO 1: ¿La RAÍZ de esta palabra es un posible cubanismo?
            if lema_lower in self.candidatos or palabra_original.lower() in self.candidatos:
                
                # PASO 2: Preguntar a BETO (IA) si en ESTA FRASE es cubanismo
                inputs = self.tokenizer(
                    texto_limpio, 
                    text_pair=palabra_original, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    prob_cub = probs[0][1].item()
                
                # Umbral de decisión
                if prob_cub > 0.35: # Bajamos un poco más para ser más sensibles
                    hallasgo = {
                        "palabra": palabra_original,
                        "confianza": f"{prob_cub*100:.1f}%",
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "lema": token.lemma_
                    }
                    hallazgos.append(hallasgo)

        if not hallazgos:
            print("\nResultado: No se detectaron cubanismos contextuales.")
        else:
            print(f"\nSe han detectado {len(hallazgos)} posibles cubanismos:")
            for h in hallazgos:
                print(f"📍 '{h['palabra']}' (Confianza: {h['confianza']})")
                print(f"   - Función: {h['pos']} / {h['dep']}")
                print(f"   - Raíz: {h['lema']}")
                print("-" * 30)

def main():
    try:
        app = DetectorCubano()
        print("\n¡Listo! El sistema está esperando tus frases.")
        
        while True:
            frase = input("\nAnalizar frase > ")
            if frase.lower() in ['salir', 'exit']: break
            if not frase.strip(): continue
            app.analizar_frase(frase)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
