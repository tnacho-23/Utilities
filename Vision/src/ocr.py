import os
import re
import shutil
from typing import List, Optional

import cv2  # <--- Added for loading saved images
import numpy as np
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from rapidocr_onnxruntime import RapidOCR

WEIGHTS_DIR = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/weights/paddle/ocr/v5"

# (hf_filename, hf_subfolder, local_name, hf_repo)
_WEIGHT_FILES = [
    ("det.onnx", "detection/v5", "v5_det.onnx", "monkt/paddleocr-onnx"),
    ("rec.onnx", "languages/english", "v5_rec_en.onnx", "monkt/paddleocr-onnx"),
    ("dict.txt", "languages/english", "v5_dict_en.txt", "monkt/paddleocr-onnx"),
    (
        "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "PP-OCRv1",
        "v1_cls.onnx",
        "SWHL/RapidOCR",
    ),
]


def ensure_weights(folder: str = WEIGHTS_DIR) -> dict:
    """Download missing OCR weights from HuggingFace into folder. Returns mapping local_name → local path."""
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for hf_filename, subfolder, local_name, repo in _WEIGHT_FILES:
        dest = os.path.join(folder, local_name)
        if not os.path.exists(dest):
            print(f"[OCR] Downloading {hf_filename} from {repo}/{subfolder} ...")
            cached = hf_hub_download(
                repo_id=repo, filename=hf_filename, subfolder=subfolder
            )
            shutil.copy2(cached, dest)
            print(f"[OCR] Saved → {dest}")
        paths[local_name] = dest
    return paths


_OLD_PLATE = re.compile(r"^[A-Z]{2}\d{4}$")  # AB1234
_NEW_PLATE = re.compile(r"^[A-Z]{4}\d{2}$")  # ABCD12
_NEW_PLATE_2026 = re.compile(r"^[A-Z]{5}\d{1}$")  # ABCDE1  (julio 2026 en adelante)
_FOREIGN_PLATE = re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$")  # AB123CD

_STRIP_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Applied only to positions that must be letters
# 0→O  1→I  2→Z  4→A  5→S  6→G  7→T  8→B
_TO_LETTER = str.maketrans("01245678", "OIZASGTB")

# Applied only to positions that must be digits
# O→0  I→1  Z→2  A→4  S→5  G→6  B→8  T→7  Q→0  D→0
_TO_DIGIT = str.maketrans("OIZASGBTQD", "0124568700")


class OcrReading(BaseModel):
    bbox: Optional[np.ndarray]  # shape (4,2) — four corner points [[x,y],...]
    text: str
    score: float

    model_config = {"arbitrary_types_allowed": True}


class Ocr:

    def __init__(
        self,
        text_score: float = 0.5,
        det_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        cls_model_path: Optional[str] = None,
        rec_keys_path: Optional[str] = None,
    ):
        kwargs = {"text_score": text_score}
        if det_model_path is not None:
            kwargs["det_model_path"] = det_model_path
        if rec_model_path is not None:
            kwargs["rec_model_path"] = rec_model_path
        if cls_model_path is not None:
            kwargs["cls_model_path"] = cls_model_path
        if rec_keys_path is not None:
            kwargs["rec_keys_path"] = rec_keys_path

        self._model = RapidOCR(**kwargs)

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, use_det=True, use_cls=True, use_rec=True)

    def forward_simple(self, img: np.ndarray) -> Optional[OcrReading]:
        """Recognition only on a pre-cropped image (no detection, no orientation)."""
        result, _ = self._model(img, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return None
        text, score = result[0]
        return OcrReading(bbox=None, text=text, score=score)

    @staticmethod
    def validate_chilean_plate(text: str, verbose: bool = False) -> Optional[str]:
        """Return the normalised plate string if it matches any Chilean format, else None."""
        clean = _STRIP_NON_ALNUM.sub("", text.upper())
        if len(clean) != 6:
            if verbose:
                print(f"[validate_plate] rejected  raw={text!r}  clean={clean!r}")
            return None

        norm1 = (
            clean[:2].translate(_TO_LETTER)
            + clean[2:4]
            + clean[4:].translate(_TO_DIGIT)
        )
        if _OLD_PLATE.match(norm1) or _NEW_PLATE.match(norm1):
            return norm1

        norm2 = clean[:5].translate(_TO_LETTER) + clean[5:].translate(_TO_DIGIT)
        if _NEW_PLATE_2026.match(norm2):
            return norm2

        if verbose:
            print(
                f"[validate_plate] rejected  raw={text!r}  norm1={norm1!r}  norm2={norm2!r}"
            )
        return None

    @staticmethod
    def validate_foreign_plate(text: str, verbose: bool = False) -> Optional[str]:
        """Return the normalised plate string if it matches the foreign format AB123CD, else None."""
        clean = _STRIP_NON_ALNUM.sub("", text.upper())
        if len(clean) != 7:
            if verbose:
                print(f"[validate_foreign] rejected  raw={text!r}  clean={clean!r}")
            return None
        norm = (
            clean[:2].translate(_TO_LETTER)
            + clean[2:5].translate(_TO_DIGIT)
            + clean[5:].translate(_TO_LETTER)
        )
        if _FOREIGN_PLATE.match(norm):
            return norm
        if verbose:
            print(f"[validate_foreign] rejected  raw={text!r}  norm={norm!r}")
        return None

    def forward_full(self, img: np.ndarray) -> List[OcrReading]:
        """Full pipeline: detection + orientation classification + recognition."""
        result, _ = self._model(img, use_det=True, use_cls=True, use_rec=True)
        if not result:
            return []
        return [
            OcrReading(bbox=np.array(box), text=text, score=score)
            for box, text, score in result
        ]


# =====================================================================
# EXAMPLE USAGE BLOCK
# =====================================================================
if __name__ == "__main__":
    # 1. Ensure weights are downloaded locally
    weight_paths = ensure_weights()

    # 2. Initialize the OCR engine with the downloaded models
    ocr_engine = Ocr(
        det_model_path=weight_paths["v5_det.onnx"],
        rec_model_path=weight_paths["v5_rec_en.onnx"],
        cls_model_path=weight_paths["v1_cls.onnx"],
        rec_keys_path=weight_paths["v5_dict_en.txt"],
    )
    ocr_engine.warmup()

    # 3. Path to your saved image (Replace with your actual file path)
    IMAGE_PATH = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/20260602_135710_track1282_crop0_none.jpg"

    if not os.path.exists(IMAGE_PATH):
        print(
            f"\n[Error] Please replace '{IMAGE_PATH}' with a real image file path."
        )
    else:
        # Load image via OpenCV
        # RapidOCR natively handles BGR arrays from cv2 smoothly
        image = cv2.imread(IMAGE_PATH)

        print(f"\n--- Processing Image: {IMAGE_PATH} ---")
        results = ocr_engine.forward_full(image)

        # 4. Print results and attempt to validate detected text as license plates
        for i, reading in enumerate(results):
            print(
                f"\n[{i}] Detected Text: {reading.text!r} (Confidence: {reading.score:.2f})"
            )

            # Try to validate as Chilean plate
            chilean_plate = Ocr.validate_chilean_plate(
                reading.text, verbose=False
            )
            if chilean_plate:
                print(f"    └ Valid Chilean Plate Found: {chilean_plate}")
                continue

            # If not Chilean, try foreign
            foreign_plate = Ocr.validate_foreign_plate(
                reading.text, verbose=False
            )
            if foreign_plate:
                print(f"    └ Valid Foreign Plate Found: {foreign_plate}")