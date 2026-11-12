"""Ghi phiên bản Python, thư viện và cấu hình máy vào results/environment.txt.

    python -m src.env_info
"""
import importlib
import os
import platform

from .config import RESULTS_DIR, SEED

LIBS = [("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"), ("scikit-image", "skimage"),
        ("imbalanced-learn", "imblearn"), ("opencv-python-headless", "cv2"), ("pillow", "PIL"),
        ("imagehash", "imagehash"), ("matplotlib", "matplotlib"), ("joblib", "joblib"),
        ("tensorflow", "tensorflow"), ("keras", "keras"), ("gradio", "gradio"),
        ("python-docx", "docx"), ("python-pptx", "pptx")]


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
             f"cpu_count {os.cpu_count()}", "gpu none (chỉ dùng CPU)", f"seed {SEED}", ""]
    for name, mod in LIBS:
        try:
            v = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as e:  # thư viện thiếu thì ghi rõ
            v = f"MISSING ({e.__class__.__name__})"
        lines.append(f"{name}=={v}")
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
        lines.insert(3, f"ram_gb {int(mem) / 1e6:.1f}")
    except Exception:
        pass
    (RESULTS_DIR / "environment.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
