"""A sentence embedder (bge, MIT) straight from transformers: the first token's vector, normalised.

The default, bge-small, is light enough for a laptop; bge-large ranks better and wants a GPU.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, name: str = DEFAULT_MODEL, batch_size: int = 32):
        from transformers import AutoModel, AutoTokenizer

        self.name, self.batch_size, self.device = name, batch_size, device()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def docs(self, texts: list[str]) -> np.ndarray:
        import torch

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=512,
                                       return_tensors="pt").to(self.device)
                pooled = self.model(**batch).last_hidden_state[:, 0]
                out[idx] = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()
        return out

    def query(self, text: str) -> np.ndarray:
        return self.docs([QUERY_PREFIX + text])[0]
