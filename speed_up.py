import cv2

# Fungsi untuk mempercepat video
def speed_up_video(input_path, output_path, speed_factor):
    # Buka video input
    cap = cv2.VideoCapture(input_path)

    # Dapatkan frame rate asli dan ukuran frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame rate baru
    new_fps = fps * speed_factor

    # Inisialisasi VideoWriter untuk menyimpan video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release semua resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path video input dan output
input_path = 'input.avi'
output_path = 'speed_up.avi'

# Faktor kecepatan (misalnya, 2.0 untuk mempercepat 2 kali)
speed_factor = 10

# Mempercepat video
speed_up_video(input_path, output_path, speed_factor)
