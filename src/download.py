"""Downloads the open dataset (Zenodo 10.5281/zenodo.11209324, ~1.4 GB), checks its MD5 and extracts it to data/.

Usage: python src/download.py
"""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://zenodo.org/records/11209324/files/data.zip?download=1"
MD5 = "ea2dd9d26ec61a5870b6455b8abe5918"


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    zpath = DATA / "data.zip"
    if not zpath.exists() or md5(zpath) != MD5:
        print("downloading", URL)
        with urllib.request.urlopen(URL) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
    if md5(zpath) != MD5:
        raise SystemExit("MD5 mismatch: the download is incomplete or the record has changed")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    # the archive may contain a top-level folder; participants must be at data/SXX
    if not (DATA / "S01").exists():
        inner = next(p for p in DATA.iterdir() if p.is_dir() and (p / "S01").exists())
        for p in inner.iterdir():
            p.rename(DATA / p.name)
    print("ok:", sorted(p.name for p in DATA.iterdir() if p.name.startswith("S")))


if __name__ == "__main__":
    main()
