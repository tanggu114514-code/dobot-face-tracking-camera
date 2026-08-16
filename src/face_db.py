"""Local LBPH face database used by the on-site enrollment channel."""

import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DB_PATH = str(DATA_ROOT / "face_dataset")
LABEL_MAP = str(DATA_ROOT / "label_mapping.json")
MODEL_SAVE = str(DATA_ROOT / "face_rec_model.yml")


os.makedirs(DB_PATH, exist_ok=True)
if not os.path.exists(LABEL_MAP):
    with open(LABEL_MAP, "w", encoding="utf-8") as file:
        json.dump({}, file)


def get_new_id():
    """Return the next numeric identity and the current label mapping."""
    with open(LABEL_MAP, "r", encoding="utf-8") as file:
        label_dict = json.load(file)
    used_ids = {int(key) for key in label_dict}
    for folder_name in os.listdir(DB_PATH):
        if folder_name.startswith("id_") and folder_name[3:].isdigit():
            used_ids.add(int(folder_name[3:]))
    if not used_ids:
        return 1, label_dict
    return max(used_ids) + 1, label_dict


def save_label_map(new_id, target_name):
    """Persist the display name for one numeric LBPH identity."""
    with open(LABEL_MAP, "r", encoding="utf-8") as file:
        label_dict = json.load(file)
    label_dict[str(new_id)] = target_name
    with open(LABEL_MAP, "w", encoding="utf-8") as file:
        json.dump(label_dict, file, ensure_ascii=False, indent=2)


def load_all_faces():
    """Load every valid 200x200 grayscale enrollment sample."""
    faces = []
    labels = []
    with open(LABEL_MAP, "r", encoding="utf-8") as file:
        label_dict = json.load(file)

    for target_id_str in label_dict:
        target_id = int(target_id_str)
        target_folder = os.path.join(DB_PATH, f"id_{target_id}")
        if not os.path.isdir(target_folder):
            continue
        for image_name in sorted(os.listdir(target_folder)):
            image_path = os.path.join(target_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (200, 200))
            faces.append(image)
            labels.append(target_id)
    return faces, np.asarray(labels, dtype=np.int32)


def train_update_model():
    """Retrain LBPH from all stored identities and save the model."""
    cv2_face = getattr(cv2, "face", None)
    creator = getattr(cv2_face, "LBPHFaceRecognizer_create", None)
    if not callable(creator):
        raise RuntimeError(
            "OpenCV LBPH is unavailable; install opencv-contrib-python"
        )

    recognizer = creator()
    all_faces, all_labels = load_all_faces()
    if not all_faces:
        print("[warning] no face samples; LBPH training skipped")
        return recognizer

    recognizer.train(all_faces, all_labels)
    recognizer.save(MODEL_SAVE)
    print(f"[learning] LBPH training complete; samples={len(all_faces)}")
    return recognizer
