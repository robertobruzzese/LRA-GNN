import cv2
import os
import pandas as pd

# Configurazione della griglia
GRID_ROWS = 6  # Numero di righe
GRID_COLS = 6  # Numero di colonne

# Percorsi delle directory
input_dir = "datasets/data/MORPH/images/Train"  # Cartella delle immagini originali
output_dir = "datasets/data/MORPH/images/Train/patched"  # Cartella per le immagini con griglia
patches_dir = "datasets/data/MORPH/patches"  # Cartella per il file CSV con le coordinate

# Creazione delle directory se non esistono
os.makedirs(output_dir, exist_ok=True)
os.makedirs(patches_dir, exist_ok=True)

# Nome del file CSV delle coordinate (valido per tutte le immagini)
csv_filename = os.path.join(patches_dir, "patches_coordinates.csv")

# Otteniamo una sola volta le coordinate della griglia
patches_data = []
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        patches_data.append([row, col])  # Salviamo solo riga e colonna

# Salviamo il file CSV delle coordinate (solo una volta)
df = pd.DataFrame(patches_data, columns=["row", "col"])
df.to_csv(csv_filename, index=False)
print(f"📂 File CSV con coordinate uniche salvato in: {csv_filename}")

# Processa tutte le immagini nella cartella di input
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Immagine {img_name} non trovata o non valida.")
        continue

    # Ottieni dimensioni dell'immagine
    height, width, _ = image.shape

    # Calcola dimensioni delle patch
    patch_height = height // GRID_ROWS
    patch_width = width // GRID_COLS

    # Disegna la griglia sull'immagine
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1, y1 = col * patch_width, row * patch_height
            x2, y2 = x1 + patch_width, y1 + patch_height

            # Disegna i bordi della patch con linee bianche
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Salva l'immagine con la griglia
    output_image_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_image_path, image)

    print(f"✅ Griglia applicata e salvata: {output_image_path}")
