#!/usr/bin/env python3
"""Evaluate generated clips: ECAPA speaker similarity + prosody metrics.

Usage: evaluate.py --files a.wav b.wav ... --ref work/refs/ref_clean.wav --out work/eval_stage1.json
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wav_16k(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav).unsqueeze(0)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t


def autocorr_f0_mean(path):
    wav, sr = sf.read(path, dtype="float64")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    win, hop = 2048, int(0.01 * sr)
    f0s = []
    for i in range(0, len(wav) - win, hop):
        fr = wav[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        fr = fr - fr.mean()
        x = fr * np.hanning(len(fr))
        nfft = 1 << int(np.ceil(np.log2(2 * len(fr))))
        X = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(X * np.conj(X), nfft)[:len(fr)]
        ac = ac / (ac[0] + 1e-12)
        a, b = int(sr / 500), min(int(sr / 70), len(fr) - 1)
        seg = ac[a:b]
        if len(seg) < 3:
            continue
        k = int(np.argmax(seg))
        if seg[k] < 0.3:
            continue
        f0s.append(sr / (a + k))
    return float(np.median(f0s)) if f0s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    from speechbrain.inference.speaker import EncoderClassifier

    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(PROJ, "models", "ecapa"),
        run_opts={"device": "cpu"},
    )

    def embed(path):
        t = load_wav_16k(path)
        with torch.no_grad():
            e = clf.encode_batch(t)
        return torch.nn.functional.normalize(e.squeeze(), dim=0)

    ref_e = embed(args.ref)
    f0_ref = autocorr_f0_mean(args.ref)

    results = []
    for f in args.files:
        e = embed(f)
        sim = float(torch.dot(ref_e, e))
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        f0 = autocorr_f0_mean(f)
        peak = float(np.max(np.abs(wav)))
        results.append({
            "file": f,
            "spk_sim": round(sim, 4),
            "dur_s": round(len(wav) / sr, 3),
            "f0_median_hz": round(f0, 1) if f0 else None,
            "f0_delta_pct": round(100 * (f0 / f0_ref - 1), 1) if (f0 and f0_ref) else None,
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        })

    results.sort(key=lambda r: -r["spk_sim"])
    report = {"ref": args.ref,
              "f0_ref_hz": round(f0_ref, 1) if f0_ref else None,
              "results": results}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
