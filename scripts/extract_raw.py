"""Extract the raw dataset archives in data/raw/ into folders next to them.

Usage: python scripts/extract_raw.py

Idempotent: an archive whose output folder already exists is skipped. Each archive is
unpacked into a temporary folder and renamed at the end, so an interrupted run never
leaves a half-extracted folder that a later run would mistake for a finished one.

Archives are extracted as-is; choosing which files belong to the label space is the job
of configs/class_mapping.yaml, not this script. The only thing dropped is PBC's
__MACOSX/ metadata folder. LISC ships as .rar, which needs 7-Zip.
"""
import os
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
# Raabin's three archives download into data/raw/raabin/ and extract beside each other
# there. Download them from
# https://dl.raabindata.com/Index%20of%20WBC%20Data/Index%20of%20WBC%20Cropped_double_labeled/
RAABIN_DIR = RAW / "raabin"


def find_rar_tool():
    """First available RAR extractor, as (flavour, executable).

    Windows usually has 7-Zip at a fixed path and nothing on PATH; a fresh Linux box
    (a rented GPU instance, say) usually has neither until p7zip-full or unrar is
    installed. Both are handled so this script does not need editing per machine.
    """
    for name in ("7z", "7za", "7zz", "unrar", "bsdtar"):
        found = shutil.which(name)
        if found:
            return ("7z" if name.startswith("7z") else name), found
    for candidate in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(candidate).exists():
            return "7z", candidate
    sys.exit(
        "no RAR extractor found. Install one of:\n"
        "  Windows: 7-Zip (https://7-zip.org)\n"
        "  Debian/Ubuntu: apt-get install p7zip-full  (or unrar)\n"
        "  macOS: brew install p7zip"
    )


def unzip_without_macosx(archive, dest):
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest, members=[m for m in z.namelist() if not m.startswith("__MACOSX/")])


def unrar(archive, dest):
    flavour, executable = find_rar_tool()
    dest.mkdir(parents=True, exist_ok=True)  # unrar and bsdtar need the target to exist
    if flavour == "7z":
        command = [executable, "x", "-y", f"-o{dest}", str(archive)]
    elif flavour == "unrar":
        command = [executable, "x", "-o+", str(archive), f"{dest}{os.sep}"]
    else:  # bsdtar; handles rar only where libarchive was built with support for it
        command = [executable, "-xf", str(archive), "-C", str(dest)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def extract(archive, top_folder, unpack, parent=None):
    """Extract `archive`, expecting it to contain a single folder named `top_folder`.

    `parent` is where that folder lands; Raabin's three archives share one, so they end
    up side by side under data/raw/raabin/ rather than loose in data/raw/.
    """
    parent = parent or RAW
    dest = parent / top_folder
    label = dest.relative_to(RAW).as_posix()
    if dest.exists():
        print(f"skip  {label}/ already exists")
        return
    if not archive.exists():
        sys.exit(f"missing archive: {archive}")
    tmp = parent / f".extracting_{top_folder}"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"extract {archive.name} -> {label}/")
    unpack(archive, tmp)
    (tmp / top_folder).rename(dest)
    shutil.rmtree(tmp)


def main():
    extract(PBC_ZIP, "PBC_dataset_normal_DIB", unzip_without_macosx)
    extract(LISC_RAR, "LISC", unrar)
    # Raabin-WBC, double-labeled cropped release. Note TestB.zip unpacks to "Test-B"
    # (hyphenated) while the other two match their archive names, and that the folder
    # names are what configs/class_mapping.yaml keys target_splits on.
    for archive, folder, unpack in (
        (RAABIN_DIR / "Train.rar", "Train", unrar),
        (RAABIN_DIR / "TestA.rar", "TestA", unrar),
        (RAABIN_DIR / "TestB.zip", "Test-B", unzip_without_macosx),
    ):
        extract(archive, folder, unpack, parent=RAABIN_DIR)


if __name__ == "__main__":
    main()
