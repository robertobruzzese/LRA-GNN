import os
import csv
import re
import cv2
from tqdm import tqdm

# Percorsi delle directory
IMAGE_DIR = "datasets/data/MORPH/images/train"
POINTS_DIR = "datasets/data/MORPH/points"
OUTPUT_CSV = "datasets/data/MORPH/images/Train/metadata/morph_data.csv"

# Funzione per estrarre ID, versione, sesso e età dal nome del file
def parse_filename(filename):
    match = re.search(r"^(\d+)_?(\d+)?([MF])(\d+)", filename, re.IGNORECASE)
    if match:
        file_id = match.group(1)  # ID numerico principale della persona
        version = match.group(2) if match.group(2) else "0"  # Versione della foto (default 0 se non presente)
        gender = match.group(3).upper()  # M o F
        age = int(match.group(4))  # Età
        return file_id, version, gender, age
    return None, None, None, None

# Funzione per ottenere il ROI (Region of Interest) dell'immagine
def get_roi(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None  # Se l'immagine non viene caricata
    h, w = image.shape[:2]  # Altezza e larghezza reali
    roi_x, roi_y = int(w * 0.1), int(h * 0.1)  # 10% del bordo
    roi_w, roi_h = int(w * 0.8), int(h * 0.8)  # 80% del centro
    return roi_x, roi_y, roi_w, roi_h

# Creazione del file CSV
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_id", "version", "sesso", "età", "roi_x", "roi_y", "roi_w", "roi_h", "img_path", "landmarks_path"])

    # Scansiona tutte le immagini
    for image_file in tqdm(os.listdir(IMAGE_DIR), desc="Elaborazione immagini"):
        if not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        image_path = os.path.join(IMAGE_DIR, image_file)
        file_id, version, gender, age = parse_filename(image_file)
        
        if file_id is None:
            print(f"⚠️ Nome file non valido: {image_file}, saltato.")
            continue
        
        # Trova il file dei landmarks
        pts_path = os.path.join(POINTS_DIR, os.path.splitext(image_file)[0] + ".pts")
        if not os.path.exists(pts_path):
            print(f"⚠️ File .pts non trovato per {image_file}, saltato.")
            continue

        # Estrai il ROI
        roi_x, roi_y, roi_w, roi_h = get_roi(image_path)

        # Scrivi nel CSV
        writer.writerow([str(file_id).zfill(5), version, gender, age, roi_x, roi_y, roi_w, roi_h, image_path, pts_path])

print(f"✅ Dati salvati in {OUTPUT_CSV}")