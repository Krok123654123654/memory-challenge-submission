# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.15.0", "transformers==4.57.6", "sentence-transformers==5.7.0",
#   "huggingface_hub==0.36.2", "numpy", "nltk", "fast-bunkai==0.1.1",
#   "sentencepiece", "protobuf",
# ]
# ///
"""Private GPU evidence preparation. No Reactor key or answer labels are used."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from huggingface_hub import HfApi, hf_hub_download

repo, revision, expected_sha, run_name = sys.argv[1:]
token = os.environ["HF_TOKEN"]
bundle = hf_hub_download(repo, "openprovence-reactor-bundle.tar.gz", repo_type="dataset",
                         revision=revision, token=token)
if hashlib.sha256(Path(bundle).read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("Bundle hash mismatch")
work = Path("/tmp/openprovence-evidence")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(bundle) as archive:
    archive.extractall(work, filter="data")
out = work / "outputs"
out.mkdir()
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable")
(out / "job.json").write_text(json.dumps({"bundle_sha256": expected_sha,
    "input_revision": revision, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}, indent=2))
code = subprocess.call([sys.executable, "examples/prepare_openprovence.py", "--input", str(work / "input"),
                        "--out", str(out), "--device", "cuda", "--save-audit"], cwd=work / "reactor",
                       env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
(out / "exit-status.json").write_text(json.dumps({"return_code": code}))
if (out / "cache").exists():
    with tarfile.open(out / "embedding-cache.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(out / "cache", arcname="cache")
HfApi(token=token).upload_folder(repo_id=repo, repo_type="dataset", folder_path=out,
                                path_in_repo=f"evidence/{run_name}", ignore_patterns=["cache/**"])
print(f"Saved evidence/{run_name}", flush=True)
sys.exit(code)
