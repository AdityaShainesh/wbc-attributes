"""Extract the raw dataset archives in data/raw/ into folders next to them.

Usage: python scripts/extract_raw.py

Idempotent: an archive whose output folder already exists is skipped. Each archive is
unpacked into a temporary folder and renamed at the end, so an interrupted run never
leaves a half-extracted folder that a later run would mistake for a finished one.

Archives are extracted as-is; choosing which files belong to the label space is the job
of configs/class_mapping.yaml, not this script. The only thing dropped is PBC's
__MACOSX/ metadata folder. LISC ships as .rar, which needs 7-Zip.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

PBC_ZIP = (
    RAW
    / "A dataset for microscopic peripheral blood cell images for development of automatic recognition systems"
    / "PBC_dataset_normal_DIB.zip"
)
LISC_RAR = RAW / "LISC.rar"


def find_7z():
    for candidate in (
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("7-Zip not found; install it to extract LISC.rar")


def unzip_without_macosx(archive, dest):
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest, members=[m for m in z.namelist() if not m.startswith("__MACOSX/")])


def unrar(archive, dest):
    subprocess.run([find_7z(), "x", "-y", f"-o{dest}", str(archive)], check=True, stdout=subprocess.DEVNULL)


def extract(archive, top_folder, unpack):
    dest = RAW / top_folder
    if dest.exists():
        print(f"skip  {top_folder}/ already exists")
        return
    if not archive.exists():
        sys.exit(f"missing archive: {archive}")
    tmp = RAW / f".extracting_{top_folder}"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"extract {archive.name} -> {top_folder}/")
    unpack(archive, tmp)
    (tmp / top_folder).rename(dest)
    shutil.rmtree(tmp)


def main():
    extract(PBC_ZIP, "PBC_dataset_normal_DIB", unzip_without_macosx)
    extract(LISC_RAR, "LISC", unrar)


if __name__ == "__main__":
    main()
