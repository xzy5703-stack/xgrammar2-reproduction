"""Fetch and verify only pinned, public research inputs; never execute them."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import requests

ROOT = Path(__file__).resolve().parent


def ensure_file(path, url, expected, verify_only):
    if not path.exists():
        if verify_only:
            raise FileNotFoundError(path.relative_to(ROOT))
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Downloaded SHA-256 mismatch: " + path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Local SHA-256 mismatch: " + path.name)
    print("Verified", path.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "source_manifest.json").read_text())
    data = json.loads((ROOT / "inputs/data_provenance.json").read_text())
    archive = ROOT / "inputs/xgrammar-v0.2.0.zip"
    ensure_file(archive, source["engine_archive_url"], source["engine_archive_sha256"], args.verify_only)
    if not args.verify_only:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe source archive member")
            handle.extractall(ROOT / "vendor")
    for item in data["files"]:
        ensure_file(ROOT / item["local_path"], item["url"], item["sha256"], args.verify_only)
    print("Pinned source archive and BFCL data verified. Run download_model.py for model assets.")


if __name__ == "__main__":
    main()
