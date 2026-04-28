import sqlite3
import csv
import sys

def export_to_csv(db_path="diccionario_cubanismos.db", csv_path="muestra_cubanismos.csv"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtener nombres de las columnas
        cursor.execute("PRAGMA table_info(cubanismos)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Obtener los datos (limitado a los primeros 500 para una revisión ágil, o todos si prefieres)
        cursor.execute("SELECT * FROM cubanismos LIMIT 500")
        rows = cursor.fetchall()
        
        # Escribir a CSV con codificación UTF-8 para soportar ñ y tildes
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';') # Usa punto y coma para no confundir con comas en las definiciones
            writer.writerow(columns)
            writer.writerows(rows)
            
        print(f"Éxito: Se exportaron {len(rows)} registros a '{csv_path}'.")
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error de base de datos: {e}")
    except Exception as e:
        print(f"Error general: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        export_to_csv(csv_path=sys.argv[1])
    else:
        export_to_csv()
