"""Local LLM inference backend (llama.cpp via llama-cpp-python) with s390x/big-endian handling.

* Default and only enabled backend: local GGUF inference. Remote LLM backends are disabled unless `llm.allow_remote: true` AND the
  data are synthetic/permitted; none is implemented here on purpose (no clinical text leaves the machine).
* GGUF files from Hugging Face are little-endian. On a big-endian host (s390x) they must be converted once
  (`gguf-convert-endian`); `ensure_native_gguf` detects the mismatch from the file header and raises a clear error.
"""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .data_loader import ROOT

_LOCK = threading.Lock()
_CACHE: dict[str, "LlamaCppBackend"] = {}


def host_byteorder() -> str:
    return sys.byteorder


def gguf_byteorder(path: Path) -> str:
    """'little' or 'big' from the GGUF header (magic 'GGUF', then uint32 version, which is 3 in both byte orders)."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    v_le = struct.unpack("<I", head[4:8])[0]
    return "little" if 0 < v_le < 100 else "big"


def ensure_native_gguf(path: Path) -> Path:
    bo = gguf_byteorder(path)
    if bo != host_byteorder():
        raise RuntimeError(f"{path.name} is {bo}-endian but this host is {host_byteorder()}-endian. Convert a copy once with: "
                           f"python -m gguf.scripts.gguf_convert_endian <copy.gguf> {host_byteorder()}  (answer YES)")
    return path


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s > 0 else 0.0


class LlamaCppBackend:
    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 2, seed: int = 0):
        from llama_cpp import Llama
        self.model_path = ensure_native_gguf(Path(model_path))
        self.name = self.model_path.name
        t = time.perf_counter()
        self.llm = Llama(model_path=str(self.model_path), n_ctx=n_ctx, n_threads=n_threads, seed=seed, verbose=False)
        self.load_s = time.perf_counter() - t
        self.size_bytes = self.model_path.stat().st_size
        self.n_threads = n_threads

    def generate_json(self, messages: list[dict], schema: dict, max_tokens: int = 500, temperature: float = 0.0) -> LLMResult:
        with _LOCK:                           # llama.cpp contexts are not thread-safe
            t = time.perf_counter()
            out = self.llm.create_chat_completion(messages=messages, response_format={"type": "json_object", "schema": schema},
                                                  temperature=temperature, max_tokens=max_tokens)
            dt = time.perf_counter() - t
        u = out["usage"]
        ch = out["choices"][0]
        return LLMResult(ch["message"]["content"], u["prompt_tokens"], u["completion_tokens"], dt, ch.get("finish_reason"))

    def generate_text(self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.0) -> LLMResult:
        with _LOCK:
            t = time.perf_counter()
            out = self.llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
            dt = time.perf_counter() - t
        u = out["usage"]
        return LLMResult(out["choices"][0]["message"]["content"], u["prompt_tokens"], u["completion_tokens"], dt, out["choices"][0].get("finish_reason"))


def get_backend(cfg: dict, which: str = "primary") -> LlamaCppBackend:
    """Cached local backend. which = 'primary' | 'large' (model names in config.yaml; big-endian copies are *.be.gguf on s390x)."""
    llm = cfg["llm"]
    name = llm[which]
    base = ROOT / llm["models_dir"]
    suffix = ".be.gguf" if host_byteorder() == "big" else ".gguf"
    path = base / (name.replace(".gguf", suffix)) if host_byteorder() == "big" else (base / "original_le" / name if (base / "original_le" / name).exists() else base / name)
    if path.name not in _CACHE:
        _CACHE[path.name] = LlamaCppBackend(path, llm["n_ctx"], llm["n_threads"], cfg["seed"])
    return _CACHE[path.name]


def get_remote_backend(cfg: dict):
    if not cfg["llm"].get("allow_remote", False):
        raise PermissionError("remote LLM backends are disabled (llm.allow_remote=false); clinical text must stay on this machine")
    raise NotImplementedError("no remote backend is implemented; add one only with explicit authorisation and permitted (synthetic) data")


class CachedBackend:
    """Disk cache around a backend: identical (model, messages, schema, max_tokens) calls reuse the stored completion AND its originally measured latency.
    Used so extractor variants that share the same first generation (B, C, C2, D, E) do not pay for it repeatedly; reported latencies stay those of the real run."""

    def __init__(self, backend, path):
        import hashlib as _h
        self._h = _h
        self.backend, self.path = backend, Path(path)
        self.name, self.size_bytes, self.n_threads, self.load_s = backend.name, backend.size_bytes, backend.n_threads, backend.load_s
        self.cache: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in open(self.path):
                r = json.loads(line)
                self.cache[r["key"]] = r

    def _key(self, messages, schema, max_tokens):
        return self._h.sha256(json.dumps([self.name, messages, schema, max_tokens], sort_keys=True).encode()).hexdigest()

    def generate_json(self, messages, schema, max_tokens: int = 500, temperature: float = 0.0) -> LLMResult:
        k = self._key(messages, schema, max_tokens)
        if k in self.cache:
            self.hits += 1
            r = self.cache[k]
            return LLMResult(r["text"], r["pt"], r["ct"], r["lat"], r.get("fr"), {"cached": True})
        self.misses += 1
        res = self.backend.generate_json(messages, schema, max_tokens, temperature)
        rec = {"key": k, "text": res.text, "pt": res.prompt_tokens, "ct": res.completion_tokens, "lat": res.latency_s, "fr": res.finish_reason, "model": self.name}
        self.cache[k] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return res

    def generate_text(self, *a, **k):
        return self.backend.generate_text(*a, **k)
