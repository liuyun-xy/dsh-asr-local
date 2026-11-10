#!/usr/bin/env python3
"""Download the three models this server needs (≈500 MB total).

    python3 dl.py [target_dir]        # default: ./models

Weights come from the official sherpa-onnx GitHub releases. If github.com is unreachable
or too slow from where you are (common in some networks), set a mirror prefix:

    ASR_MIRROR=https://gh-proxy.com/ python3 dl.py

Only `model.int8.onnx` / `tokens.txt` / `tokens.json` are kept — the paraformer tarball also
carries a 786 MB fp32 model we never load, and it is deleted as soon as it is unpacked.
"""
import os
import sys
import tarfile
import time
import urllib.request

MIRROR = os.environ.get("ASR_MIRROR", "")
DEST = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "models")

RELEASES = {
    "paraformer": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                  "sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2",
    "punct": "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
             "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
}
VAD_CANDIDATES = [
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
]


def fetch(url, path, tries=4):
    for attempt in range(tries):
        try:
            t0, got = time.time(), 0
            req = urllib.request.Request(MIRROR + url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as r, open(path, "wb") as f:
                while True:
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (100 << 20) < (1 << 18):
                        el = max(time.time() - t0, 1)
                        print(f"  {got / 1048576:6.0f} MB  {got / 1048576 / el:5.1f} MB/s", flush=True)
            return got
        except Exception as e:
            print(f"  retry {attempt + 1}/{tries}: {type(e).__name__}: {e}", flush=True)
            time.sleep(4)
    raise RuntimeError(f"download failed: {url}")


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, url in RELEASES.items():
        tgz = os.path.join(DEST, f"{name}.tar.bz2")
        if not (os.path.exists(tgz) and os.path.getsize(tgz) > (100 << 20)):
            print(f"{name}: downloading…", flush=True)
            fetch(url, tgz)
        print(f"{name}: {os.path.getsize(tgz) / 1048576:.0f} MB — unpacking", flush=True)
        with tarfile.open(tgz, "r:bz2") as t:
            keep = [m for m in t.getmembers()
                    if m.name.endswith(("model.int8.onnx", "model.onnx", "tokens.txt", "tokens.json"))]
            t.extractall(DEST, members=keep)
        os.remove(tgz)

    vad = os.path.join(DEST, "silero_vad.onnx")
    if not os.path.exists(vad):
        for url in VAD_CANDIDATES:
            try:
                print(f"vad: downloading {url.rsplit('/', 1)[-1]}", flush=True)
                fetch(url, vad, tries=2)
                break
            except Exception as e:
                print(f"vad: {e}", flush=True)
    if os.path.exists(vad) and os.path.getsize(vad) < 100_000:
        os.remove(vad)
    print(f"done → {DEST}", flush=True)
    if not os.path.exists(vad):
        print("! silero_vad.onnx missing — download it manually and set ASR_VAD_MODEL", flush=True)


if __name__ == "__main__":
    main()
