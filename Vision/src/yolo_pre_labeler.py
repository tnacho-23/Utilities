import os
import argparse
from ultralytics import YOLO
import cv2

def auto_label(input_path, output_path, model_path, conf_threshold=0.25):
    """
    Loads a YOLO model, runs inference on a folder of images, and saves
    the detections in YOLO format (.txt) to an output directory.
    """
    # Load the model
    model = YOLO(model_path)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Supported image extensions
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(input_path) if f.lower().endswith(img_extensions)]

    print(f"Found {len(images)} images. Starting auto-labeling...")

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(output_path, label_name)
        
        result = results[0]

        # Check if there are any detections
        if len(result.boxes) > 0:
            # Save the actual detections
            result.save_txt(label_path)
        else:
            # Create an empty file for background images
            open(label_path, 'a').close() 
            print(f"No detections for {img_name}, created empty label.")
        
        # Optional: Copy the image to the output folder as well if needed
        # import shutil
        # shutil.copy(img_path, os.path.join(output_path, img_name))

    print(f"Processing complete. Labels saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using YOLO weights.")
    parser.add_argument("--input", type=str, default="C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/images/fup_patente_easton_p2", required=False, help="Path to folder containing images.")
    parser.add_argument("--output", type=str, default="C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/labels/fup_patente_easton_p2", required=False, help="Path to save generated .txt labels.")
    parser.add_argument("--weights", type=str, default="C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/weights/FUP_mall_y26mV1.pt", required=False, help="Path to .pt YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")

    args = parser.parse_args()

    auto_label(args.input, args.output, args.weights, args.conf)