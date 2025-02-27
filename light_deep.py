from ultralytics import YOLO
import cv2

# Load model YOLOv8 Nano (paling kecil dan cepat)
model = YOLO('yolov8n.pt')

# Buka video input
video_path = '/Users/owwl/Downloads/object_counting/video5.mp4'
cap = cv2.VideoCapture(video_path)

# Periksa apakah video input bisa dibuka
if not cap.isOpened():
    print("Error: Gagal membuka video input. Periksa path file.")
    exit(1)

# Dapatkan properti video untuk persiapan output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Turunkan resolusi video (opsional)
scale_percent = 50  # Turunkan resolusi menjadi 50%
frame_width = int(frame_width * scale_percent / 100)
frame_height = int(frame_height * scale_percent / 100)

# Buat VideoWriter untuk menyimpan output
output_path = '/Users/owwl/Downloads/object_counting/output_video5_2frames.mp4'
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec untuk format MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Counter untuk membatasi hanya 24 frame
frame_count = 0
max_frames = 24

# Loop melalui setiap frame dalam video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Hanya proses 24 frame pertama
    if frame_count >= max_frames:
        break

    # Turunkan resolusi frame
    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # Lakukan detection dan tracking menggunakan model YOLOv8
    results = model.track(frame, persist=True, half=True, classes=[0])  # Hanya lacak orang (class ID 0)

    # Visualisasi hasil detection dan tracking
    annotated_frame = results[0].plot()

    # Tulis frame yang telah di-annotasi ke output video
    out.write(annotated_frame)

    # Tampilkan frame yang telah di-annotasi (opsional)
    cv2.imshow('YOLOv8 Tracking - 24 Frames', annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame_count += 1
    print(f"Frame {frame_count} diproses")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video (24 frame pertama) disimpan di: {output_path}")