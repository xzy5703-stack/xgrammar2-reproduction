"""Download a fixed public checkpoint, without account tokens or remote model code."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    meta = json.loads((ROOT / "source_manifest.json").read_text())
    manifest_path = ROOT / "model_manifest.json"
    expected = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if expected and (expected["revision"] != meta["model_revision"] or expected["repo"] != meta["model_repo"]):
        raise ValueError("Source and expected model manifest disagree")
    patterns = ["*.json", "merges.txt", "vocab.json", "LICENSE", "README.md"]
    if not args.tokenizer_only:
        patterns.append("*.safetensors")
    target = ROOT / "models" / "Qwen3-0.6B"
    print(f"Downloading {meta['model_repo']} @ {meta['model_revision']}", flush=True)
    snapshot_download(repo_id=meta["model_repo"], revision=meta["model_revision"],
                      local_dir=str(target), allow_patterns=patterns, token=False,
                      max_workers=3, cache_dir=str(ROOT / ".hf_cache"))
    assets = {}
    for p in sorted(target.iterdir()):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            assets[p.name] = {"sha256": digest.hexdigest(), "bytes": p.stat().st_size}
            if expected and p.name in expected["assets"] and assets[p.name] != expected["assets"][p.name]:
                raise ValueError("Model asset hash mismatch: " + p.name)
    if expected and not args.tokenizer_only:
        if set(expected["assets"]) - set(assets):
            raise ValueError("Expected model asset is missing")
    manifest_path.write_text(json.dumps({
        "repo": meta["model_repo"], "revision": meta["model_revision"],
        "license": meta["model_license"], "assets": assets,
    }, indent=2) + "\n")
    print("Verified local assets:", ", ".join(assets), flush=True)


if __name__ == "__main__":
    main()
