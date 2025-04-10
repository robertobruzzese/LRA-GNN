import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm

# 📌 Percorsi delle directory
IMAGE_DIR = "datasets/data/FGNET/images/Train/images_preprocessed"
POINTS_DIR = "datasets/data/FGNET/points"
REDUCED_POINTS_DIR = "datasets/data/FGNET/points_reduced"


# ✅ Assicuriamoci che le directory di output esistano
os.makedirs(POINTS_DIR, exist_ok=True)
os.makedirs(REDUCED_POINTS_DIR, exist_ok=True)

# 📌 Carichiamo il rilevatore di volti e il predittore di landmark di dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("/Users/robertobruzzese/Desktop/LRA-GNN/datasets/shape_predictor_68_face_landmarks.dat")

# 📌 Lista dei landmark chiave per l'invecchiamento
KEY_LANDMARKS = [
    30, 31, 35, 36, 39, 42, 45, 48, 54, 62,  # Punti originali
    27, 28, 29, 32, 33, 34, 37, 38, 40, 41,  # Nuovi punti vicini (naso e occhi)
    43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57  # Punti per occhi e bocca
]

# 📌 Dimensione target (immagine preprocessata)
TARGET_SIZE = (224, 224)

def save_landmarks(image_path, points_path, reduced_points_path):
    """Estrae i landmark, li normalizza e li salva"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossibile leggere l'immagine: {image_path}")
        return
    
    height, width = image.shape[:2]  # 📌 Otteniamo la vera dimensione dell'immagine
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        print(f"⚠️ Nessun volto trovato in {image_path}. Salto immagine.")
        return
    
    face = faces[0]  # Se più volti, prendiamo solo il primo
    landmarks = landmark_predictor(gray, face)
    
    if landmarks.num_parts < 68:
        print(f"⚠️ Attenzione: solo {landmarks.num_parts} landmark trovati in {image_path}.")
        return

    # 🔹 Estraggo i landmark come numpy array
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    # ✅ **Normalizziamo i landmark rispetto alla dimensione reale dell'immagine**
    points[:, 0] = (points[:, 0] / width) * TARGET_SIZE[0]  # Normalizza X
    points[:, 1] = (points[:, 1] / height) * TARGET_SIZE[1]  # Normalizza Y

    # 🔴 **Salva i landmark completi**
    with open(points_path, "w") as f:
        f.write("version: 1\n")
        f.write("n_points: 68\n")
        f.write("{\n")
        for x, y in points:
            f.write(f"{x:.2f} {y:.2f}\n")
        f.write("}\n")
    
    print(f"✅ Landmarks normalizzati e salvati in {points_path}")

    # 🔵 **Seleziona solo i landmark chiave**
    reduced_points = points[KEY_LANDMARKS]

    # ✅ **Salva i landmark ridotti**
    with open(reduced_points_path, "w") as f:
        f.write("version: 1\n")
        f.write(f"n_points: {len(reduced_points)}\n")
        f.write("{\n")
        for x, y in reduced_points:
            f.write(f"{x:.2f} {y:.2f}\n")
        f.write("}\n")

    print(f"✅ Landmarks ridotti normalizzati e salvati in {reduced_points_path}")

# 🔄 **Elaboriamo tutte le immagini nella cartella preprocessata**
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    points_path = os.path.join(POINTS_DIR, os.path.splitext(image_file)[0] + ".pts")
    reduced_points_path = os.path.join(REDUCED_POINTS_DIR, os.path.splitext(image_file)[0] + ".pts")
    
    save_landmarks(image_path, points_path, reduced_points_path)
