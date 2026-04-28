import sqlite3
import os

DB_PATH = "diccionario_cubanismos.db"

def init_db():
    print(f"Inicializando base de datos en {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Crear la tabla principal
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cubanismos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lema TEXT NOT NULL,
        categoria_gramatical TEXT,
        funcion_sintactica TEXT,
        marcas_dialectales TEXT,
        definicion TEXT NOT NULL,
        ejemplo TEXT,
        fuente TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Base de datos y tabla 'cubanismos' creadas correctamente.")

def check_pdf(pdf_path):
    print(f"\nRevisando {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: No se encontró el archivo {pdf_path}")
        return
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        print(f"Total de páginas: {len(doc)}")
        
        # Revisar las primeras 3 páginas para ver si tienen texto
        text_found = False
        for i in range(min(3, len(doc))):
            page = doc.load_page(i)
            text = page.get_text()
            if text.strip():
                text_found = True
                print(f"Página {i+1} tiene texto (Muestra: {text[:50].replace(chr(10), ' ')}...)")
            else:
                print(f"Página {i+1} parece ser una imagen (no hay texto extraíble directamente).")
                
        if text_found:
            print("Conclusión: El PDF contiene texto vectorizado. Podemos usar PyMuPDF.")
        else:
            print("Conclusión: El PDF parece ser un conjunto de imágenes escaneadas. Necesitaremos OCR (pytesseract).")
            
    except ImportError:
        print("Falta instalar PyMuPDF. Por favor ejecuta 'pip install pymupdf'")
    except Exception as e:
        print(f"Error al procesar el PDF: {e}")

if __name__ == "__main__":
    init_db()
    check_pdf("CUBA1.pdf")
    check_pdf("CUBA2.pdf")
