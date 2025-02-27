import cv2

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        return -1
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return total_frames

if __name__ == "__main__":
    video_path = "WhatsApp Video 2025-02-27 at 13.17.41.mp4"
    frames = count_frames(video_path)
    
    if frames > 0:
        print(f"Total number of frames in the video: {frames}")
