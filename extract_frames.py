import cv2
import os

def extract_nth_frames(video_path, output_dir, n=10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Save frame if it's a multiple of n
        if frame_count % n == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "WhatsApp Video 2025-02-27 at 13.17.41.mp4"
    output_dir = "extracted_frames"
    extract_nth_frames(video_path, output_dir, 18)
