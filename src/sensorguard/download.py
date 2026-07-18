"""Download and verify the official UCI AI4I archive."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
ARCHIVE_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
ARCHIVE_MEMBER = "ai4i2020.csv"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(destination: str | Path, *, force: bool = False) -> Path:
    """Download, checksum, and extract the dataset to destination."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    csv_path = destination_path / ARCHIVE_MEMBER
    if csv_path.exists() and not force:
        return csv_path

    archive_path = destination_path / "ai4i-601.zip"
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual_hash = file_sha256(archive_path)
    if actual_hash != ARCHIVE_SHA256:
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"dataset checksum mismatch: expected {ARCHIVE_SHA256}, got {actual_hash}")
    with zipfile.ZipFile(archive_path) as archive:
        member_names = set(archive.namelist())
        if member_names != {ARCHIVE_MEMBER}:
            raise ValueError(f"unexpected archive members: {sorted(member_names)}")
        archive.extract(ARCHIVE_MEMBER, destination_path)
    archive_path.unlink(missing_ok=True)
    return csv_path

