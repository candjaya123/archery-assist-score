import cv2
import numpy as np

def detect_green_circles(image_path, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    # Baca gambar
    image = cv2.imread(image_path)

    # Konversi gambar ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisikan batasan warna hijau di HSV
    lower_green = np.array([min_hue, min_saturation, min_value])
    upper_green = np.array([max_hue, max_saturation, max_value])

    # Thresholding untuk mendapatkan hanya warna hijau
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Cari kontur di masker
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    # Iterasi melalui setiap kontur yang ditemukan
    for contour in contours:
        # Hitung momen kontur
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            # Gambar lingkaran dan pusat kontur (opsional untuk visualisasi)
            cv2.circle(image, (cX, cY), 10, (0, 255, 0), -1)
        else:
            cX, cY = 0, 0

    # Tampilkan gambar (opsional untuk visualisasi)
    cv2.imshow("Detected Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return centers

# Contoh penggunaan fungsi
image_path = 'image2.png'
min_hue = 48
max_hue = 48
min_saturation = 150
max_saturation = 150
min_value = 173
max_value = 173

centers = detect_green_circles(image_path, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)
print(centers)
print("Pusat dari lingkaran hijau yang terdeteksi:", centers)
