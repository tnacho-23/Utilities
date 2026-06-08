import os
from pathlib import Path
from ultralytics import YOLO
# Import the built-in classification dataset splitter
from ultralytics.data.split import split_classify_dataset

def train_with_auto_split():
    # 1. Path to your raw, unsplit dataset
    # This folder should just contain subfolders for each class.
    # Example: raw_dataset/cats/*.jpg, raw_dataset/dogs/*.jpg
    raw_dataset_path = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/DCP/DCP_bodega"
    
    if not os.path.exists(raw_dataset_path):
        raise ValueError(f"The path '{raw_dataset_path}' does not exist.")

    # 2. Automatically split the dataset
    # This creates a new folder named 'raw_dataset_split' with an 80/20 train/val split.
    print("Splitting dataset automatically...")
    split_dataset_path = split_classify_dataset(raw_dataset_path, train_ratio=0.8)
    print(f"Dataset successfully split! Saved at: {split_dataset_path}")

    # 3. Load the pretrained YOLO26 Classifier
    print("Loading YOLO Classifier...")
    model = YOLO("yolo11m-cls.pt") 

    # 4. Train using the newly created split path
    print("Starting training...")
    results = model.train(
        data=str(split_dataset_path), # Pass the automatically generated path
        epochs=400,
        imgsz=480,
        batch=32,
        workers=4,
        device=0,                     # Use 0 for GPU, or 'cpu'
        project="Door",
        name="train_door"
    )
    
    print("Training complete!")

if __name__ == "__main__":
    train_with_auto_split()