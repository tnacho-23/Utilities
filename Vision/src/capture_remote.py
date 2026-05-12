import cv2
import os
import json

# Configuration
JSON_FILE = "gamma2.json"
OUTPUT_FOLDER = "captures"

# Ensure output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load cameras from JSON
try:
    with open(JSON_FILE, 'r') as f:
        STREAMS = json.load(f)
except FileNotFoundError:
    print(f"Error: {JSON_FILE} not found.")
    exit()

for cam in STREAMS:
    name = cam.get("name", "Unknown_Camera")
    url = cam.get("url")
    
    if not url:
        print(f"Skipping {name}: No URL provided.")
        continue

    print(f"Connecting to {name}...")
    cap = cv2.VideoCapture(url)
    
    # Optional: Set a timeout or check if camera is opened
    if not cap.isOpened():
        print(f"Failed to open stream for {name}")
        continue

    ret, frame = cap.read()
    
    if ret:
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale and thickness adjusted for typical 1080p streams
        cv2.putText(frame, name, (30, 60), font, 1.8, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Sanitize filename (remove characters that might break file systems)
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).replace(' ', '_')
        output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.jpg")
        
        cv2.imwrite(output_path, frame)
        print(f"Success: Saved {output_path}")
    else:
        print(f"Failed: Could not read frame from {name}")
        
    cap.release()

print("\nCapture process complete.")