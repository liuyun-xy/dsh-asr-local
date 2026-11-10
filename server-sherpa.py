#!/usr/bin/env python3
"""sherpa-onnx ASR server — OpenAI-compatible `/v1/audio/transcriptions`.

Speaks the contract dsh-video-storyboard expects (see its docs/asr-contract.md):

    GET  /health                    → {"status":"ok"}
    GET  /v1/models                 → model list
    POST /v1/audio/transcriptions   → verbose_json { text, model, language, segments:[{start,end,text}] }
    POST /shutdown                  → replies, then exits (for asr.json manage.idleMs)

Pipeline: silero VAD segments the speech → paraformer-zh transcribes each segment →
ct-transformer adds punctuation. Everything runs locally; nothing leaves the machine.

Env:
    ASR_MODEL_DIR   where the model folders live        (default: ./models)
    ASR_PORT        listen port                          (default: 8000)
    ASR_THREADS     threads per model                    (default: 4)
    ASR_MODEL_NAME  name reported back to the client      (default: paraformer-zh-2024-03-09)
    ASR_VAD_MODEL   explicit path to silero_vad.onnx      (default: auto-detect)
"""
import io
import json
import os
import signal
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sherpa_onnx

ROOT = os.path.abspath(os.environ.get("ASR_MODEL_DIR", "models"))
PORT = int(os.environ.get("ASR_PORT", "8000"))
THREADS = int(os.environ.get("ASR_THREADS", "4"))
MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "paraformer-zh-2024-03-09")


def find(substr: str, filename: str):
    """Shallow model discovery: first folder whose path contains `substr` and holds `filename`."""
    hits = []
    for base, _dirs, files in os.walk(ROOT):
        if substr in base and filename in files:
            hits.append(os.path.join(base, filename))
    return sorted(hits, key=len)[0] if hits else None


def load():
    asr_model = find("paraformer", "model.int8.onnx") or find("paraformer", "model.onnx")
    if not asr_model:
        sys.exit(f"no paraformer model under {ROOT}\n"
                 f"run: python3 dl.py   (or set ASR_MODEL_DIR)")
    tokens = os.path.join(os.path.dirname(asr_model), "tokens.txt")

    punct_model = find("punct", "model.onnx")

    vad_model = os.environ.get("ASR_VAD_MODEL") or find("", "silero_vad.onnx")
    if not vad_model or not os.path.exists(vad_model):
        sys.exit(f"no silero_vad.onnx under {ROOT} — run dl.py or set ASR_VAD_MODEL")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=asr_model, tokens=tokens, num_threads=THREADS,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search", debug=False,
    )

    punctuator = None
    if punct_model:
        try:
            punctuator = sherpa_onnx.OfflinePunctuation(
                sherpa_onnx.OfflinePunctuationConfig(
                    model=sherpa_onnx.OfflinePunctuationModelConfig(
                        ct_transformer=punct_model, num_threads=THREADS, debug=False)))
        except Exception as e:  # punctuation is cosmetic — never fatal
            print(f"punctuation disabled: {e}", flush=True)

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    cfg.silero_vad.min_silence_duration = float(os.environ.get("ASR_MIN_SILENCE", "0.4"))
    cfg.silero_vad.min_speech_duration = float(os.environ.get("ASR_MIN_SPEECH", "0.25"))
    cfg.sample_rate = 16000

    return recognizer, punctuator, cfg, asr_model, vad_model


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, rate, sw, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"expect 16-bit PCM wav, got sample width {sw}")
    if rate != 16000:
        raise ValueError(f"expect 16kHz, got {rate}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def extract_wav(raw: bytes, ctype: str) -> bytes:
    """Pull the `file` part out of a multipart body (or accept a bare wav)."""
    if "multipart/form-data" not in ctype:
        return raw
    boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
    for part in raw.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        if b"filename=" in head and body:
            return body.rsplit(b"\r\n", 1)[0]
    raise ValueError("no file part in multipart body")


print("loading models…", flush=True)
_t0 = time.time()
recognizer, punctuator, vad_config, ASR_PATH, VAD_PATH = load()


def recognize(samples) -> str:
    if samples is None or len(samples) < 1600:  # < 0.1 s
        return ""
    s = recognizer.create_stream()
    s.accept_waveform(16000, samples)
    recognizer.decode_stream(s)
    text = (s.result.text or "").strip()
    if text and punctuator is not None:
        try:
            text = punctuator.add_punctuation(text)
        except Exception:  # cosmetic only
            pass
    return text.strip()


def transcribe(x: np.ndarray):
    """VAD → per-segment decode. Contract: start/end are relative to the uploaded audio."""
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
    segs, win = [], 512  # silero wants 512 samples per feed
    for i in range(0, len(x), win):
        chunk = x[i:i + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        vad.accept_waveform(chunk)
        while not vad.empty():
            seg = vad.front
            t = recognize(seg.samples)
            if t:
                segs.append({"start": round(seg.start / 16000, 3),
                             "end": round(seg.start / 16000 + len(seg.samples) / 16000, 3),
                             "text": t})
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        t = recognize(seg.samples)
        if t:
            segs.append({"start": round(seg.start / 16000, 3),
                         "end": round(seg.start / 16000 + len(seg.samples) / 16000, 3),
                         "text": t})
        vad.pop()
    if not segs:  # VAD found nothing: fall back to the whole clip
        whole = recognize(x)
        if whole:
            segs = [{"start": 0.0, "end": round(len(x) / 16000, 3), "text": whole}]
    return sorted(segs, key=lambda s: s["start"])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[asr] " + (fmt % args) + "\n")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p in ("/health", "/v1/models"):
            self._json(200, {"status": "ok", "model": MODEL_NAME,
                             "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.rstrip("/")
        if p.endswith("/shutdown"):
            self._json(200, {"status": "bye"})
            threading.Thread(target=lambda: (time.sleep(0.2), os.kill(os.getpid(), signal.SIGTERM)),
                             daemon=True).start()
            return
        if not p.endswith("/audio/transcriptions"):
            self._json(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            x = read_wav(extract_wav(raw, self.headers.get("Content-Type", "")))
            t = time.time()
            segs = transcribe(x)
            sys.stderr.write(f"[asr] {len(x) / 16000:.1f}s → {len(segs)} seg in {time.time() - t:.1f}s\n")
            self._json(200, {"text": "".join(s["text"] for s in segs),
                             "model": MODEL_NAME, "language": "zh", "segments": segs})
        except Exception as e:  # contract: a failure must really be non-2xx
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"ready in {time.time() - _t0:.1f}s\n  asr : {ASR_PATH}\n  vad : {VAD_PATH}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    print(f"listening on http://127.0.0.1:{PORT}/v1", flush=True)
    srv.serve_forever()
