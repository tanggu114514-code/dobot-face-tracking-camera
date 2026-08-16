from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_private_runtime_files_are_absent(self):
        forbidden = [
            ROOT / "data" / "face_rec_model.yml",
            ROOT / "data" / "label_mapping.json",
        ]
        for path in forbidden:
            self.assertFalse(path.exists(), f"private runtime file present: {path}")

    def test_enrollment_directories_contain_no_images(self):
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
        for folder in (ROOT / "data" / "targets", ROOT / "data" / "face_dataset"):
            images = [
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in image_suffixes
            ]
            self.assertEqual(images, [], f"enrollment images present: {images}")

    def test_public_default_is_simulation(self):
        config_text = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
        self.assertIn("SIMULATION_MODE = True", config_text)

    def test_no_local_windows_user_paths(self):
        source_suffixes = {".py", ".md", ".json", ".txt"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in source_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "C:\\Users\\" in text:
                offenders.append(path)
        self.assertEqual(offenders, [], f"local user paths present: {offenders}")


if __name__ == "__main__":
    unittest.main()
