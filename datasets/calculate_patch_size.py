import cv2
import os

# Percorso della cartella contenente le immagini
input_dir = "datasets/data/MORPH/images/Train"

# Configurazione della griglia
GRID_ROWS = 6  # Numero di righe
GRID_COLS = 6  # Numero di colonne

# Trova la prima immagine valida nella cartella
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not image_files:
    print(f"⚠️ Nessuna immagine trovata in {input_dir}. Controlla il percorso.")
else:
    img_path = os.path.join(input_dir, image_files[0])  # Prende la prima immagine trovata
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Immagine {image_files[0]} non valida.")
    else:
        # Ottieni dimensioni dell'immagine
        height, width, _ = image.shape
        print(f"📏 Dimensioni dell'immagine '{image_files[0]}': {width}×{height} pixel")

        # Calcola dimensioni di una patch
        patch_width = width // GRID_COLS
        patch_height = height // GRID_ROWS
        print(f"🟩 Dimensioni di una patch: {patch_width}×{patch_height} pixel")
