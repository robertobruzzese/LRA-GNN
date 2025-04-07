import os
from utils.preprocessing import preprocess_image
 # importa la funzione se è in un altro file
# oppure, se la funzione è nello stesso script, copiala direttamente qui sopra

# === Percorsi delle cartelle ===
input_dir = "datasets/data/MORPH/images/Train"
output_dir = os.path.join(input_dir, "images_preprocessed")
os.makedirs(output_dir, exist_ok=True)  # Crea la cartella di output se non esiste

# === Estrai tutti i file immagine .jpg nella cartella input ===
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]

print(f"🔍 Trovate {len(image_files)} immagini da preprocessare...")

# === Ciclo su tutte le immagini ===
for idx, image_name in enumerate(image_files, 1):
    image_path = os.path.join(input_dir, image_name)
    preprocessed_image_path = os.path.join(output_dir, image_name)

    try:
        print(f"\n🖼️ [{idx}/{len(image_files)}] Preprocessing: {image_name}")
        preprocess_image(image_path, preprocessed_image_path)
    except Exception as e:
        print(f"⚠️ Errore durante il preprocessing di {image_name}: {e}")

print("\n✅ Preprocessing batch completato.")
