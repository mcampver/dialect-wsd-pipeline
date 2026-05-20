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

# Añadimos un pequeño utilitario offline para evitar instalar modelos de HF solo para el sentimiento
import sys
BASE_APP_DIR = os.path.dirname(__file__)
WEBAPP_DIR = os.path.join(BASE_APP_DIR, "..", "09_webapp")
sys.path.append(os.path.abspath(WEBAPP_DIR))
try:
    from utils_nlp import analizar_sentimiento_basico
except ImportError:
    # Fallback por si ejecutas el script fuera de su contexto
    analizar_sentimiento_basico = lambda x: "Neutro"

# Configuración de rutas
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
        self.candidatos_traduccion = {}
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                # Intentamos extraer también la traducción, si no existe la columna omitiremos
                try:
                    df_cubs = pd.read_sql_query("SELECT lema, traduccion FROM cubanismos", conn)
                    for index, row in df_cubs.iterrows():
                        lema = row['lema'].lower().strip()
                        trad = row['traduccion']
                        if pd.isna(trad): trad = "Sin equivalente registrado"
                        self.candidatos_traduccion[lema] = trad
                except sqlite3.OperationalError:
                    # Si no existe la columna 'traduccion', hacemos fallback a listado simple
                    df_cubs = pd.read_sql_query("SELECT DISTINCT lema FROM cubanismos", conn)
                    for l in df_cubs['lema'].unique():
                        self.candidatos_traduccion[l.lower()] = "Sin traducción"
                
                conn.close()
                print(f"Diccionario cargado: {len(self.candidatos_traduccion)} términos cubanos reconocidos.")
            except Exception as e:
                print(f"Error cargando DB, usando lista de respaldo. ({e})")
        
        # Diccionario de respaldo offline
        respaldo = {
            "asere": "amigo/socio", "acere": "amigo/socio", 
            "guagua": "autobús", "máquina": "automóvil antiguo", "maquina": "automóvil antiguo",
            "bola": "rumor", "pinchar": "trabajar", "finca": "prisión",
            "ómnibus": "autobús", "que bola": "hola / qué tal"
        }
        for k, v in respaldo.items():
            if k not in self.candidatos_traduccion:
                self.candidatos_traduccion[k] = v

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
            traduccion_estandar = self.candidatos_traduccion.get(lema_lower, "Desconocido")
            if traduccion_estandar == "Desconocido" and palabra_original.lower() in self.candidatos_traduccion:
                traduccion_estandar = self.candidatos_traduccion.get(palabra_original.lower())

            if traduccion_estandar != "Desconocido":
                
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
                    sentimiento = analizar_sentimiento_basico(texto_limpio)
                    
                    hallasgo = {
                        "palabra": palabra_original,
                        "confianza": round(prob_cub * 100, 1),
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "lema": token.lemma_,
                        "equivalente": traduccion_estandar,
                        "sentimiento": sentimiento
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
                
        return hallazgos

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
