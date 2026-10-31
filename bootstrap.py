"""Fetch the pinned organizer kit locally; do not relicense or vendor it."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for item in lock["files"]:
        target = ROOT / item["path"]
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest == item["installed_sha256"]:
                continue
            if digest != item["sha256"]:
                raise RuntimeError(f"Refusing to overwrite modified {item['path']}")
        else:
            data = urllib.request.urlopen(item["url"], timeout=30).read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Upstream hash mismatch: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if item["path"] == "reactor/client.py":
            patch = ROOT / "patches/reactor-client.patch"
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=ROOT, check=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["installed_sha256"]:
            raise RuntimeError(f"Installed hash mismatch: {item['path']}")
    print("Pinned organizer files installed.")


if __name__ == "__main__":
    main()
