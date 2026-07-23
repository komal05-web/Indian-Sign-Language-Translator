"""
startup.py
==========
Downloads ISL model files from Google Drive if they don't exist locally.
Called from build.sh before the server starts.

HOW TO USE:
  1. Upload your .keras and .json files to Google Drive
  2. Right-click each file → Share → Anyone with link → Copy link
  3. Extract the file ID from the link:
     https://drive.google.com/file/d/FILE_ID_HERE/view
  4. Paste the FILE_ID into the dict below
"""

import os
import urllib.request

FILES = {
    # "destination_path": "google_drive_file_id"
    "word/isl_best_model.keras":           os.getenv("WORD_MODEL_ID", ""),
    "word/label_map.json":                 os.getenv("LABEL_MAP_ID", ""),
    "sentence/isl_sentence_model.keras":   os.getenv("SENT_MODEL_ID", ""),
    "sentence/sentence_label_map.json":    os.getenv("SENT_LABEL_ID", ""),
}

def download_file(file_id, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    print(f"[Startup] Downloading → {dest_path} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size = os.path.getsize(dest_path)
        print(f"[Startup] Done  ({size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"[Startup] FAILED: {e}")

def main():
    for dest, file_id in FILES.items():
        if not file_id:
            print(f"[Startup] Skipping {dest} — no file ID set")
            continue
        if os.path.exists(dest):
            print(f"[Startup] Already exists: {dest}")
            continue
        download_file(file_id, dest)

if __name__ == "__main__":
    main()