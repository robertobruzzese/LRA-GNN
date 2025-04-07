import cv2

preprocessed_image_path = "datasets/data/MORPH/images/Train/images_preprocessed/01_0M32.jpg"

image = cv2.imread(preprocessed_image_path)
if image is None:
    print("❌ ERRORE: L'immagine non è stata trovata!")
else:
    height, width = image.shape[:2]
    print(f"✅ Dimensioni dell'immagine: {width}x{height}")
