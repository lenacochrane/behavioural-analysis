
import cv2

# Load the video
video_path = '/Users/cochral/Desktop/PLUG-VIDEOS/untitled folder/2024-07-11_16-45-27_pc107_2024_07_09_kir2.1.mp4'
output_path = '/Users/cochral/Desktop/PLUG-VIDEOS/untitled folder/2CND.mp4'

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Get the video frame rate and size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define text for different time ranges
time_ranges = [
    (0, 1, "Embryo"),  # From 0s to 1s
    (1, 28, "First Instar Larvae"),   # From 2s to 10s
    (29, 56, "Second Instar Larvae"), # From 10s to 15s
    (57,120, "Third Instar Larvae")
    # Add more time ranges if needed
]

# Font settings for the text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_color = (255, 255, 255)  # White text
thickness = 3
position = (100, 1300)  # Position where the text will be placed

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the current time in seconds
    current_time = frame_count / fps

    # Check which time range the current frame belongs to and add corresponding text
    for start, end, text in time_ranges:
        if start <= current_time < end:
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
            break  # No need to check other ranges if one is matched

    # Write the frame with or without text
    out.write(frame)

    # Update the frame count
    frame_count += 1

# Release everything once the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved with text overlay.")
