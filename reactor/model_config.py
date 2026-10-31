"""Pinned, permissively licensed models used by the active retrieval backend."""
import json
from pathlib import Path

_LOCK = json.loads((Path(__file__).with_name("models.lock.json")).read_text())
ACTIVE = _LOCK["active_retrieval"]
RETRIEVAL_PROFILE = ACTIVE["profile"]
EMBEDDER_PIN = next(m for m in _LOCK["models"] if m["id"] == ACTIVE["embedding"])
RERANKER_PIN = next(m for m in _LOCK["models"] if m["id"] == ACTIVE["reranker"])
PRUNING_THRESHOLD = ACTIVE.get("pruning_threshold")


def require_current_profile(root):
    """Do not silently combine cached evidence or trials from different rerankers."""
    path = Path(root) / "protocol.json"
    if path.exists():
        profile = json.loads(path.read_text()).get("retrieval_profile")
        if profile != RETRIEVAL_PROFILE:
            raise ValueError("This experiment has a different or unversioned retrieval pipeline. "
                             "Prepare a new output directory for the active retrieval pipeline.")
