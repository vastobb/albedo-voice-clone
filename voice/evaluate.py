#!/usr/bin/env python3
"""Evaluate generated clips:

1. ASR phrase check (Whisper, local) — does the take actually say the phrase?
2. ECAPA speaker similarity vs the reference.
3. Prosody metrics (F0, duration, peak).

Usage: evaluate.py --files a.wav b.wav ... --ref work/refs/ref_clean.wav \
                   --out work/eval_stage1.json --asr-model models/whisper-small
"""
import argparse
import difflib
import json
import os
import unicodedata

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = {
    "ja": "あらあら、さよなら。",
    "en": "Ara ara... sayonara.",
}
TOKENS = {
    "ja": ["あら", "さよなら"],
    "en": ["ara", "sayonara"],
}


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # katakana block
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(s: str, lang: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    if lang == "ja":
        s = kata_to_hira(s)
        return "".join(ch for ch in s if ("\u3041" <= ch <= "\u3096") or ch.isalnum())
    return "".join(ch for ch in s.lower() if ch.isalnum())


def asr_score(transcript: str, target: str, lang: str) -> float:
    t = normalize_text(transcript, lang)
    n = normalize_text(target, lang)
    if not t or not n:
        return 0.0
    ratio = difflib.SequenceMatcher(None, t, n).ratio()
    tokens = TOKENS.get(lang, [])
    hits = sum(1 for tok in tokens if tok in t) / len(tokens) if tokens else 0.0
    return round(0.5 * ratio + 0.5 * hits, 4)


def guess_lang(path: str) -> str:
    base = os.path.basename(path)
    for lang in ("ja", "en"):
        if f"_{lang}_" in base:
            return lang
    return "ja"


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
    ap.add_argument("--asr-model", default="openai/whisper-small")
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)

    # ---- ASR phrase check
    transcripts = {}
    if not args.no_asr:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=args.asr_model)
        lang_name = {"ja": "japanese", "en": "english"}
        for f in args.files:
            lang = guess_lang(f)
            try:
                r = asr(f, generate_kwargs={"language": lang_name.get(lang, "japanese"),
                                            "task": "transcribe"})
                transcripts[f] = r["text"].strip()
            except Exception as e:  # never let ASR kill the pipeline
                transcripts[f] = ""
                print(f"ASR_ERR {f}: {type(e).__name__} {e}")
            print(f"ASR {os.path.basename(f)}: {transcripts[f]}", flush=True)

    # ---- ECAPA speaker similarity
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
        lang = guess_lang(f)
        transcript = transcripts.get(f, "")
        r = {
            "file": f,
            "spk_sim": round(sim, 4),
            "dur_s": round(len(wav) / sr, 3),
            "f0_median_hz": round(f0, 1) if f0 else None,
            "f0_delta_pct": round(100 * (f0 / f0_ref - 1), 1) if (f0 and f0_ref) else None,
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        }
        if not args.no_asr:
            r["transcript"] = transcript
            r["asr_score"] = asr_score(transcript, TARGETS.get(lang, ""), lang)
        results.append(r)

    results.sort(key=lambda x: -x["spk_sim"])
    report = {"ref": args.ref,
              "f0_ref_hz": round(f0_ref, 1) if f0_ref else None,
              "results": results}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
