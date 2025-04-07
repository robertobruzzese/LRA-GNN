import os
import cv2
import numpy as np
from tqdm import tqdm

# Percorsi delle directory
IMAGE_DIR = "datasets/data/MORPH/images/Train/images_preprocessed/"  # Cartella con le immagini originali
POINTS_DIR = "datasets/data/MORPH/points"        # Cartella con i file .pts originali (68 punti)
REDUCED_POINTS_DIR = "datasets/data/MORPH/points_reduced"  # Cartella con i file .pts ridotti (PCA)
OUTPUT_DIR_FULL = "datasets/data/MORPH/images/Train/landmarked"  # Per le immagini con 68 landmark
OUTPUT_DIR_REDUCED = "datasets/data/MORPH/images/Train/landmarked_reduced"  # Per i landmark ridotti

# Crea le cartelle di output se non esistono
os.makedirs(OUTPUT_DIR_FULL, exist_ok=True)
os.makedirs(OUTPUT_DIR_REDUCED, exist_ok=True)

# Funzione per leggere i punti da un file .pts
def read_landmarks(pts_path):
    with open(pts_path, "r") as f:
        lines = f.readlines()
        points = []
        reading = False
        for line in lines:
            line = line.strip()
            if line == "{":
                reading = True
                continue
            elif line == "}":
                break
            if reading:
                try:
                    x, y = map(float, line.split())  # Converte in float
                    points.append((x, y))  # Mantiene valori float per normalizzazione successiva
                except ValueError:
                    print(f"⚠️ Errore nel parsing del file: {pts_path}, linea: {line}")
                    continue
        return points


# Funzione per normalizzare i landmark ridotti
def normalize_landmarks(landmarks, img_width, img_height):
    if not landmarks:
        return []

    landmarks = np.array(landmarks)

    # Trasla i punti per evitare valori negativi
    min_x, min_y = np.min(landmarks, axis=0)
    landmarks[:, 0] -= min_x
    landmarks[:, 1] -= min_y

    # Normalizza e scala nei limiti dell'immagine
    max_x, max_y = np.max(landmarks, axis=0)
    
    if max_x > 0:
        landmarks[:, 0] = (landmarks[:, 0] / max_x) * img_width
    if max_y > 0:
        landmarks[:, 1] = (landmarks[:, 1] / max_y) * img_height

    # Arrotonda e converte in interi
    return [(int(round(x)), int(round(y))) for x, y in landmarks]


# Processa ogni immagine nella cartella
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for image_file in tqdm(image_files, desc="Drawing landmarks"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    
    # Percorsi ai file .pts
    pts_path_full = os.path.join(POINTS_DIR, os.path.splitext(image_file)[0] + ".pts")
    pts_path_reduced = os.path.join(REDUCED_POINTS_DIR, os.path.splitext(image_file)[0] + ".pts")

    # Percorsi per salvare le immagini con i landmark disegnati
    output_path_full = os.path.join(OUTPUT_DIR_FULL, image_file)
    output_path_reduced = os.path.join(OUTPUT_DIR_REDUCED, image_file)

    if not os.path.exists(pts_path_full):
        print(f"❌ File .pts non trovato per {image_file}, saltato.")
        continue

    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Errore nel caricamento dell'immagine: {image_path}")
        continue

    # 🔴 Disegna i landmark completi (68 punti)
    landmarks_full = read_landmarks(pts_path_full)
    for (x, y) in landmarks_full:
        cv2.circle(image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)  # ROSSO

    # Salva l'immagine con i landmark completi
    cv2.imwrite(output_path_full, image)

    # 🔵 Disegna i landmark ridotti solo se il file esiste
    if os.path.exists(pts_path_reduced):
        image_reduced = cv2.imread(image_path)  # Usa l'immagine originale
        landmarks_reduced = read_landmarks(pts_path_reduced)

        print(f"📍 Landmark ridotti per {image_file}: {landmarks_reduced}") 

        print(f"🔍 {image_file}: {len(landmarks_reduced)} landmark ridotti dopo normalizzazione")

        # Disegna i landmark ridotti
        for (x, y) in landmarks_reduced:
            cv2.circle(image_reduced, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)  # BLU

        
        # Salva l'immagine con i landmark ridotti
        cv2.imwrite(output_path_reduced, image_reduced)
        print(f"✅ Salvata immagine ridotta: {output_path_reduced}")

    
print(f"✅ Immagini con 68 landmark salvate in: {OUTPUT_DIR_FULL}")
print(f"✅ Immagini con landmark ridotti salvate in: {OUTPUT_DIR_REDUCED}")
