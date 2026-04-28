"""
zip_dataset.py
Utilidad para empaquetar el dataset BIO y los scripts de entrenamiento 
para subirlo a Google Colab.
"""

import os
import zipfile

def main():
    # Incluir el nuevo dataset BIO además del original
    folders_to_zip = ['dataset_bio_cubanismos', 'dataset_cubanismos_hf']
    output_filename = 'tesis_cubanismos_colab.zip'
    scripts_to_include = ['train_multi.py', 'train_transformer.py', 'convertir_bio.py']
    
    print(f"📦 Empaquetando datasets y scripts...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Añadir carpetas de datasets
        for folder in folders_to_zip:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        full_path = os.path.join(root, file)
                        zipf.write(full_path, full_path)
            else:
                print(f"⚠️ Carpeta no encontrada: {folder}")
        
        # Añadir scripts en la raíz del zip
        for script in scripts_to_include:
            if os.path.exists(script):
                zipf.write(script, script)
            else:
                print(f"⚠️ Script no encontrado: {script}")
    
    print(f"✅ Archivo '{output_filename}' creado con éxito.")
    print("\nSube este archivo a Google Colab, descomprímelo y ejecuta:")
    print("  !python convertir_bio.py    # Si no tienes el dataset BIO")
    print("  !python train_multi.py      # Para el entrenamiento NER")

if __name__ == "__main__":
    main()
